import os
import os.path as osp
import numpy as np

import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric import seed_everything

from typing import Optional, Union, Any
from explainers import gnnexplainer, ig, deconvolution
from tqdm import tqdm

def ws_graph_model(
        N = 500,
        k = 4,
        p = 0.001,
        seed = None):
    if seed is not None:
        seed_everything(seed)
    G = nx.watts_strogatz_graph(N, k, p, seed=seed)
    A = nx.to_numpy_array(G)
    N, E = G.number_of_nodes(), G.number_of_edges()
    return G, A, N, E

def sbm_graph_model(
        n_blocks = 3,
        avg_block_size = 50,
        block_size_dev = 2,
        mu = 0.01,
        sigma = 0.001,
        diag_mu = 0.8,
        diag_sigma = 0.1,
        seed=None
        ):
    if seed is not None:
        seed_everything(seed)
    sizes = np.random.randint(avg_block_size-block_size_dev, high=avg_block_size+block_size_dev, size=n_blocks)
    probs = np.random.normal(mu, scale=sigma, size=(n_blocks, n_blocks))
    np.fill_diagonal(probs, np.random.normal(diag_mu, scale=diag_sigma, size=n_blocks))
    probs = np.triu(probs) + np.triu(probs, k=1).T
    probs = np.clip(probs, 0, 1)
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    A = nx.to_numpy_array(G)

    N, E = G.number_of_nodes(), G.number_of_edges()

    node_block_labels = nx.get_node_attributes(G, 'block')

    del G.graph['partition']  # don't know where it comes from
    return G, A, N, E, node_block_labels


def to_networkx_simple(
        data: 'torch_geometric.data.Data',
        node_names: Optional[list] = [],
        to_undirected: Optional[Union[bool, str]] = False
    ) -> Any:
    

    G = nx.Graph() if to_undirected else nx.DiGraph()

    if node_names:
        G.add_nodes_from(node_names)
    else:
        G.add_nodes_from(range(data.num_nodes))

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        G.add_edge(u, v)

    return G

def get_computation_graph_as_nx(
        source_node,
        target_node,
        data,
        num_hops = 2
        ):
    subset, edge_index, mapping, edge_mask = k_hop_subgraph([source_node, target_node], num_hops, data.edge_index)
    computation_graph = Data(data.x[subset, :], data.edge_index[:, edge_mask])
    computation_graph = to_networkx_simple(computation_graph, node_names=list(subset.numpy()), to_undirected=True)
    return computation_graph

def normalize_bounds(v, u = 1., l = 0.5):
    vmin = np.min(v)
    vmax = np.max(v)
    return np.array([l+(x-vmin)*(u-l)/(vmax-vmin) for x in v])

def load_curves(graph_model, model_name, explainer, feat_type, tot_range, target=1, seed=0):
    path = f"../outputs/{graph_model}/{model_name}/{explainer}/curves/"
    print(graph_model, model_name, explainer, feat_type, tot_range, target)
    deletions = []   
    for i in range(tot_range):
        for fname in os.listdir(path):
            if fname.endswith(f'{feat_type}_curve.npy') and fname.split('_')[0] == str(seed) and fname.split('_')[1] == str(i):
                if target == 1:
                    right_pred = float(fname.split('_')[-5])>0.5
                elif target == 0:
                    right_pred = float(fname.split('_')[-5])<0.5
                else:
                    right_pred = True
                if right_pred:
                    with open(osp.join(path, fname), 'rb') as f:
                        del_curve = np.load(f)
                    deletions.append(del_curve)
    print(len(deletions))
    return deletions

def get_explanation(
    explainer,
    model,
    train_data,
    edge_label_index,
    **kwargs
    ):
    if explainer == 'gnnexplainer':
        explanation = gnnexplainer(
            model, 
            train_data.x, 
            train_data.edge_index, 
            edge_label_index, 
            return_type=kwargs['return_type']
            )
    if explainer == 'ig':
        explanation = ig(
                        model,
                        train_data.x, 
                        train_data.edge_index, 
                        edge_label_index
                    )
    if explainer == 'deconvolution':
        explanation = deconvolution(
            model,
            train_data.x, 
            train_data.edge_index, 
            edge_label_index
        )

    edge_mask = explanation['edge_mask'].numpy()
    node_mask = explanation['node_mask'].numpy()
    return explanation, edge_mask, node_mask
