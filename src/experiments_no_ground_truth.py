import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from torch_geometric import seed_everything

import numpy as np
import pandas as pd

import networkx as nx
from torch_geometric.utils.convert import from_networkx

from tqdm import tqdm

from models import LinkGIN, LinkGCN, LinkSAGE, DeepVGAE
from decoders import InnerProductDecoder, CosineDecoder
from metrics import deletion_curve_edges, deletion_curve_features
from utils import get_explanation
from torch_geometric.utils import k_hop_subgraph
from matplotlib import pyplot as plt
import seaborn as sb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_model = 'cora'
model_name = 'gcn'
explainer = 'random'
decoder = 'inner'
seed = 0

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

if explainer == 'random':
    sorting = 'random'
else:
    sorting = 'descending'
    
print(seed, graph_model, model_name, explainer, decoder, return_type)  

output_folder = f"../outputs/{graph_model}/{model_name}/{explainer}/"

for seed in range(1):
    print(seed, graph_model, model_name, explainer, decoder, return_type)  

    seed_everything(seed)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True),
    ])

    dataset = graph_model.capitalize()
    path = osp.join('../', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset, transform=transform)
    train_data, val_data, test_data = dataset[0]

    seed_everything(seed)
    if model_name == 'gin':
        model = LinkGIN(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 61
    if model_name == 'gcn':
        model = LinkGCN(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 201
    if model_name == 'sage':
        model = LinkSAGE(train_data.num_features, 128, 64, sim=decoder).to(device)
        tot_epochs = 101
    if model_name == 'vgae':
        if decoder == 'inner':
            model = DeepVGAE(train_data.num_features, 128, 64, InnerProductDecoder()).to(device)
        if decoder == 'cosine':
            model = DeepVGAE(train_data.num_features, 128, 64, CosineDecoder()).to(device)
        tot_epochs = 1501

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
            
    results = []
    for i in tqdm(range(val_data.edge_label_index.size(1))[:100]):
        edge_label_index = val_data.edge_label_index[:, [i]]
        source_node, target_node = edge_label_index.numpy()[:, 0]

        target = val_data.edge_label[i].item()
        if sigmoid:
            pred = model(train_data.x, train_data.edge_index, edge_label_index).sigmoid().item()
        else:
            pred = model(train_data.x, train_data.edge_index, edge_label_index).item()
        if target==1 and int(pred>0.5)==target:
            source_node, target_node, pred, target
            
            if explainer != 'random':
                explanation, edge_mask, node_mask = get_explanation(
                        explainer, model, train_data, edge_label_index, return_type=return_type
                    )

                edge_mask = explanation['edge_mask'].numpy()
                node_mask = explanation['node_mask'].numpy()
            else:
                edge_mask = np.zeros(train_data.edge_index.size(1))
                node_mask = np.zeros(train_data.x.size())
            
            subset, sub_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
                [source_node, target_node], 
                2, 
                train_data.edge_index)

            #  Edges
            deletion_curve = deletion_curve_edges(
                model,
                train_data.x,
                sub_edge_index,
                edge_label_index,
                edge_mask[sub_edge_mask],
                sigmoid=sigmoid,
                sorting=sorting
            )

            output_path = osp.join(output_folder, 'curves',
                            f"{seed}_{i}_{source_node}_{target_node}_{pred:.3f}_{target}_{test_auc:.3f}_{decoder}_edge_deletion_curve.npy") 
            with open(output_path, 'wb') as f:
                np.save(f, deletion_curve)

            #  Features / Nodes
            feature_mean_mask = node_mask.mean(axis=0)
            node_mean_mask = node_mask.mean(axis=1)
            feature_base_values = train_data.x.mean(dim=0)

            feature_deletion_curve = deletion_curve_features(
                model,
                train_data.x,
                sub_edge_index,
                edge_label_index,
                feature_mean_mask,
                feature_base_values,
                sigmoid=sigmoid,
                sorting=sorting
            )

            output_path = osp.join(output_folder, 'curves',
                            f"{seed}_{i}_{source_node}_{target_node}_{pred:.3f}_{target}_{test_auc:.3f}_{decoder}_feature_deletion_curve.npy") 
            with open(output_path, 'wb') as f:
                np.save(f, feature_deletion_curve)
