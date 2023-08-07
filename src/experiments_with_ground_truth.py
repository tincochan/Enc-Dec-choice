import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T

from torch_geometric import seed_everything

import numpy as np
import pandas as pd

import networkx as nx
from torch_geometric.utils.convert import from_networkx

from tqdm import tqdm

from models import LinkGIN, LinkGCN, LinkSAGE, DeepVGAE
from decoders import InnerProductDecoder, CosineDecoder
from explainers import gnnexplainer, ig, deconvolution, backprop
from metrics import ws_confusion_matrix, sbm_confusion_matrix, sensitivity_specificity
from utils import ws_graph_model, sbm_graph_model, get_computation_graph_as_nx
from utils import get_explanation
from plotting import visualize_explanation

from matplotlib import pyplot as plt
import seaborn as sb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_model = 'ws'
model_name = 'sage'
explainer = 'ig'
decoder = 'inner'
binary_threshold = 0.4

if model_name == 'vgae':
    sigmoid = False
else:
    sigmoid = True

if model_name == 'vgae':
    return_type = 'probs'
    from train_test import train_vgae as train
    from train_test import test_vgae as test
else:
    return_type = 'raw'
    from train_test import train
    from train_test import test

output_folder = f"../outputs/{graph_model}/{model_name}/{explainer}/"

for seed in range(10):
    print(seed, graph_model, model_name, explainer, decoder, return_type, binary_threshold)  

    seed_everything(seed)
    if graph_model == 'ws':
        N = 500
        k = 4
        p = 0.001
        G, A, N, E = ws_graph_model(N=N, k=k, p=p, seed=seed)
    if graph_model == 'sbm':
        n_blocks = 3
        avg_block_size = 50
        block_size_dev = 2
        mu = 0.01
        sigma = 0.001
        diag_mu = 0.8
        diag_sigma = 0.1
        G, A, N, E, node_block_labels = sbm_graph_model(seed=seed)

    seed_everything(seed)
    X = np.eye(N).astype(np.float32)
    data = from_networkx(G)
    data.x = torch.tensor(X)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True),
    ])

    train_data, val_data, test_data = transform(data)

    seed_everything(seed)
    if model_name == 'gin':
        model = LinkGIN(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 41
    if model_name == 'gcn':
        model = LinkGCN(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 51
    if model_name == 'sage':
        model = LinkSAGE(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 101
    if model_name == 'vgae':
        if decoder == 'inner':
            model = DeepVGAE(train_data.num_features, 128, 64, InnerProductDecoder()).to(device)
        if decoder == 'cosine':
            model = DeepVGAE(train_data.num_features, 128, 64, CosineDecoder()).to(device)
        tot_epochs = 1001
        
    if graph_model == 'sbm':
        tot_epochs = 41

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    seed_everything(0)
    for epoch in range(1, tot_epochs):
        loss = train(model, optimizer, train_data)
        if epoch % 20 == 0:
            if model_name == 'vgae':
                val_auc = test(model, train_data, val_data)
                test_auc = test(model, train_data, test_data)
            else:
                val_auc = test(model, val_data)
                test_auc = test(model, test_data)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                f'Test: {test_auc:.4f}')
            
    tpr_tnr_results = []
    for i in tqdm(range(val_data.edge_label_index.size(1))):
        edge_label_index = val_data.edge_label_index[:, [i]]
        source_node, target_node = edge_label_index.numpy()[:, 0]

        target = val_data.edge_label[i].item()
        if sigmoid:
            pred = model(train_data.x, train_data.edge_index, edge_label_index).sigmoid().item()
        else:
            pred = model(train_data.x, train_data.edge_index, edge_label_index).item()
        if graph_model == 'ws':
            condition = np.abs(source_node-target_node)<(k/2+1)
        if graph_model == 'sbm':
            condition = node_block_labels[source_node]==node_block_labels[target_node]
        if condition and int(pred>0.5)==target:
            source_node, target_node, pred, target
            
            explanation, edge_mask, node_mask = get_explanation(
                    explainer, model, train_data, edge_label_index, return_type=return_type
                )

            edge_mask = explanation['edge_mask'].numpy()
            node_mask = explanation['node_mask'].numpy()
            
            output_path = osp.join(output_folder, 'masks',
                            f"{seed}_{i}_{test_auc:.3f}_one_hot_{decoder}_edge_mask.npy") 
            with open(output_path, 'wb') as f:
                np.save(f, edge_mask)
            output_path = osp.join(output_folder, 'masks', 
                            f"{seed}_{i}_{test_auc:.3f}_one_hot_{decoder}_node_mask.npy") 
            with open(output_path, 'wb') as f:
                np.save(f, node_mask)

            computation_graph = get_computation_graph_as_nx(source_node, target_node, train_data)

            if graph_model == 'ws':
                conf, tp, tn, fp, fn = ws_confusion_matrix(
                                        train_data.edge_index, 
                                        edge_mask, 
                                        explanation['node_mask'], 
                                        computation_graph, 
                                        source_node, 
                                        target_node, 
                                        binary_threshold
                                    )
            else:
                conf, tp, tn, fp, fn = sbm_confusion_matrix(
                                        node_block_labels,
                                        train_data.edge_index, 
                                        edge_mask, 
                                        explanation['node_mask'], 
                                        computation_graph, 
                                        source_node, 
                                        target_node, 
                                        binary_threshold
                                    )
            
            tpr, tnr = sensitivity_specificity(conf)
            
            if graph_model == 'ws':
                tpr_tnr_results.append([i, target, pred, source_node, target_node, tpr, tnr, binary_threshold])
            if graph_model == 'sbm':
                tpr_tnr_results.append([i, target, pred, source_node, target_node, tpr, tnr, binary_threshold, node_block_labels[source_node], node_block_labels[target_node]])
        
    df = pd.DataFrame(tpr_tnr_results)

    if graph_model == 'ws':
        df.columns = ['idx', 'target', 'pred', 'source_node', 'target_node', 'tpr', 'tnr', 'binary_threshold']
    if graph_model == 'sbm':
        df.columns = ['idx', 'target', 'pred', 'source_node', 'target_node', 'tpr', 'tnr', 'binary_threshold', 'source_block_label', 'target_block_label'] 
    
    df['seed'] = seed
    df['graph'] = graph_model
    if graph_model == 'ws':
        df['N'] = N
        df['k'] = k
        df['p'] = p
    if graph_model == 'sbm':
        df['n_blocks'] = n_blocks
        df['avg_block_size'] = avg_block_size
        df['diag_mu'] = diag_mu
    df['test_auc'] = test_auc
    df['model'] = model_name
    df['explainer'] = explainer
    df['decoder'] = decoder

    output_path = osp.join(output_folder, 
                        f"{seed}_{test_auc:.3f}_one_hot_{decoder}_tpr_tnr.csv")

    df.to_csv(output_path)