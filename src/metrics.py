import torch
import numpy as np

import networkx as nx

from typing import Optional, Union, Any

from tqdm import tqdm

from utils import normalize_bounds


def get_ws_tp_edges(graph, source_node, target_node):
    triangle_nodes = set(graph.neighbors(source_node)).intersection(graph.neighbors(target_node))

    triangle_edges = []
    for n in triangle_nodes:
        for m in (source_node, target_node):
            triangle_edges.append(tuple(sorted((n, m))))
            # triangle_edges.append((m, n))

    return triangle_edges

def get_ws_tn_edges(graph, source_node, target_node):
    tp_edges = get_ws_tp_edges(graph, source_node, target_node)
    n_graph = graph.copy()
    n_graph.remove_edges_from(tp_edges)

    return [tuple(sorted(e)) for e in n_graph.edges()]

def ws_confusion_matrix(
        edge_index,
        edge_mask,
        node_mask,
        graph,
        source_node,
        target_node,
        binary_threshold
        ):
    
    model_p_edges_idx = np.where(edge_mask>binary_threshold)[0]
    model_p_edges = edge_index[:, model_p_edges_idx].T.numpy()
    model_p_edges = [tuple(sorted(e)) for e in model_p_edges]

    model_n_edges = [tuple(sorted(e)) for e in graph.edges() if tuple(sorted(e)) not in model_p_edges]

    tp_edges = get_ws_tp_edges(graph, source_node, target_node)
    tn_edges = get_ws_tn_edges(graph, source_node, target_node) 
    
    tp = []
    tn = []
    fp = []
    fn = []
    for e in model_p_edges:
        if e in tp_edges:
            tp.append(e)
        else:
            fp.append(e)

    for e in model_n_edges:
        if e in tn_edges:
            tn.append(e)
        else:
            fn.append(e)

    tp = list(set(tp))
    tn = list(set(tn))
    fp = list(set(fp))
    fn = list(set(fn))

    conf = np.array([
        [len(tp), len(fp)], 
        [len(fn), len(tn)]]
        )

    return conf, tp, tn, fp, fn

def get_sbm_tp_edges(graph, source_node, target_node, node_labels):
    source_block = node_labels[source_node]
    target_block = node_labels[target_node]
    if source_block != target_block:
        return []
    else:
        inter_block_edges = []
        for s, t in graph.edges:
            if node_labels[s] == node_labels[source_node] \
                        and node_labels[t] == node_labels[source_node]:
                inter_block_edges.append(tuple(sorted((s, t)))) 
        return inter_block_edges

def get_sbm_tn_edges(graph, source_node, target_node, node_labels):
    tp_edges = get_sbm_tp_edges(graph, source_node, target_node, node_labels)
    n_graph = graph.copy()
    n_graph.remove_edges_from(tp_edges)

    return [tuple(sorted(e)) for e in n_graph.edges()]

def sbm_confusion_matrix(
        node_labels,
        edge_index,
        edge_mask,
        node_mask,
        graph,
        source_node,
        target_node,
        binary_threshold
        ):
    model_p_edges_idx = np.where(edge_mask>binary_threshold)[0]
    model_p_edges = edge_index[:, model_p_edges_idx].T.numpy()
    model_p_edges = [tuple(sorted(e)) for e in model_p_edges]

    model_n_edges = [tuple(sorted(e)) for e in graph.edges() if tuple(sorted(e)) not in model_p_edges]
    
    tp_edges = get_sbm_tp_edges(graph, source_node, target_node, node_labels)
    tn_edges = get_sbm_tn_edges(graph, source_node, target_node, node_labels) 
    
    tp = []
    tn = []
    fp = []
    fn = []
    for e in model_p_edges:
        if e in tp_edges:
            tp.append(e)
        else:
            fp.append(e)

    for e in model_n_edges:
        if e in tn_edges:
            tn.append(e)
        else:
            fn.append(e)

    tp = list(set(tp))
    tn = list(set(tn))
    fp = list(set(fp))
    fn = list(set(fn))

    conf = np.array([
        [len(tp), len(fp)], 
        [len(fn), len(tn)]]
        )

    return conf, tp, tn, fp, fn

def sensitivity_specificity(confusion_matrix):  # TPR = TP/(TP+FN) TNR = TN/(TN+FP)
    
    tp = confusion_matrix[0, 0]
    tn = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]

    if (tp+fn) == 0.:
          sensitivity = 1.
    else:
        sensitivity = tp/(tp+fn)

    if (tn+fp) == 0.:
        specificity = 1.
    else:
        specificity = tn/(tn+fp)

    return sensitivity, specificity

def get_tpr_tnr_curves(
        edge_index,
        edge_mask,
        node_mask,
        computation_graph,
        source_node,
        target_node,
        bt_range,
        graph = 'ws',
        node_block_labels = None
        ):
    tpr_tnr_curve = []
    for binary_threshold in bt_range:
        assert graph in ['ws', 'sbm']
        if graph == 'ws':
            conf, tp, tn, fp, fn = ws_confusion_matrix(
                                    edge_index, 
                                    edge_mask, 
                                    node_mask, 
                                    computation_graph, 
                                    source_node, 
                                    target_node, 
                                    binary_threshold
                                )
        if graph == 'sbm':
            conf, tp, tn, fp, fn = sbm_confusion_matrix(
                                                    node_block_labels,
                                                    edge_index, 
                                                    edge_mask, 
                                                    node_mask, 
                                                    computation_graph, 
                                                    source_node, 
                                                    target_node, 
                                                    binary_threshold
                                                )
        tpr, tnr = sensitivity_specificity(conf)
        tpr_tnr_curve.append([tpr, tnr])
    return np.array(tpr_tnr_curve)

def deletion_curve_edges(
        model,
        x,
        edge_index,
        edge_label_index,
        edge_mask,
        sorting='descending',
        trials=20,
        sigmoid=True,
        seed=None 
        ):
    if sorting=='descending':
        edge_mask_sort_idx = np.argsort(edge_mask)[::-1]
        hard_mask = torch.ones(edge_mask.shape[0], dtype=torch.long)
        if sigmoid:
            deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
        else:
            deletion_curve = [model(x, edge_index, edge_label_index).item()]
        for i in tqdm(edge_mask_sort_idx):
            hard_mask[i] = 0
            sub_edge_index = (edge_index*hard_mask).clone()
            if sigmoid:
                deletion_curve.append(model(x, sub_edge_index, edge_label_index).sigmoid().item())
            else:
                deletion_curve.append(model(x, sub_edge_index, edge_label_index).item())

    if sorting=='random':
        if seed is not None:
            np.random.seed(seed)

        deletion_curve = []    
        for _ in tqdm(range(trials)):
            edge_mask_sort_idx = list(range(edge_mask.shape[0]))
            np.random.shuffle(edge_mask_sort_idx)
            hard_mask = torch.ones(edge_mask.shape[0], dtype=torch.long)
            if sigmoid:
                single_deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
            else:
                single_deletion_curve = [model(x, edge_index, edge_label_index).item()]
            for i in edge_mask_sort_idx:
                hard_mask[i] = 0
                sub_edge_index = (edge_index*hard_mask).clone()
                if sigmoid:
                    single_deletion_curve.append(model(x, sub_edge_index, edge_label_index).sigmoid().item())
                else:
                    single_deletion_curve.append(model(x, sub_edge_index, edge_label_index).item())
            deletion_curve.append(single_deletion_curve)

    return np.array(deletion_curve)

def deletion_curve_features(
        model,
        x,
        edge_index,
        edge_label_index,
        feature_mask,
        feature_base_values,
        sorting='descending',
        trials=20,
        sigmoid=True,
        seed=None 
        ):
    if sorting=='descending':
        feature_mask_sort_idx = np.argsort(feature_mask)[::-1]
        hard_mask = x.clone()
        if sigmoid:
            deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
        else:
            deletion_curve = [model(x, edge_index, edge_label_index).item()]
        for i in tqdm(feature_mask_sort_idx):
            hard_mask[:, i] = feature_base_values[i]
            if sigmoid:
                deletion_curve.append(model(hard_mask, edge_index, edge_label_index).sigmoid().item())
            else:
                deletion_curve.append(model(hard_mask, edge_index, edge_label_index).item())

    if sorting=='random':
        if seed is not None:
            np.random.seed(seed)

        deletion_curve = []    
        for _ in tqdm(range(trials)):
            feature_mask_sort_idx = list(range(feature_mask.shape[0]))
            np.random.shuffle(feature_mask_sort_idx)
            hard_mask = x.clone()
            if sigmoid:
                single_deletion_curve = [model(x, edge_index, edge_label_index).sigmoid().item()]
            else:
                single_deletion_curve = [model(x, edge_index, edge_label_index).item()]
            for i in feature_mask_sort_idx:
                hard_mask[:, i] = feature_base_values[i]
                if sigmoid:
                    single_deletion_curve.append(model(hard_mask, edge_index, edge_label_index).sigmoid().item())
                else:
                    single_deletion_curve.append(model(hard_mask, edge_index, edge_label_index).item())
            deletion_curve.append(single_deletion_curve)

    return np.array(deletion_curve)

def random_line(n, c0, cf, x):
    return c0+(cf-c0)/x*n

def compute_upper_area(c0, cf):
    return 0.5*(2-c0-cf)

def linear_area_score(deletion_curve, normalize=False):
    if normalize:
        deletion_curve = normalize_bounds(deletion_curve)
    c0 = deletion_curve[0]
    cf = deletion_curve[-1]
    norm = deletion_curve.shape[0]
    random_baseline = np.array([random_line(n, c0, cf, norm) for n in range(norm)])
    upper_area = compute_upper_area(c0, cf)
    lower_area = 1.-upper_area
    Ap = np.maximum(deletion_curve-random_baseline, 0).sum()/norm
    Am = np.maximum(random_baseline-deletion_curve, 0).sum()/norm

    return Am/lower_area - Ap/upper_area

def area_score(deletion_curve, random_baseline, norm=True):
    assert deletion_curve.shape==random_baseline.shape
    if norm:
        deletion_curve = normalize_bounds(deletion_curve)
        random_baseline = normalize_bounds(random_baseline)
    # score = 0
    Ap = 0
    Am = 0
    l = 0
    u = 0
    for i in range(deletion_curve.shape[0]):
        Ap += np.max([deletion_curve[i]-random_baseline[i], 0])
        Am += np.max([random_baseline[i]-deletion_curve[i], 0])
        l += random_baseline[i] - np.min([np.min(deletion_curve), np.min(random_baseline)])
        u += 1. - random_baseline[i] 
        # score += random_baseline[i]-deletion_curve[i]
    # return score/(deletion_curve.shape[0]-2)
    print(f"Am {Am:.2f}, Ap {Ap:.2f}, l {l:.2f}, u {u:.2f}, Am/l {Am/l:.2f}, Ap/u {Ap/u:.2f}")
    return Am/l - Ap/u

def fidelity_sparsity(deletion_curve, normalize=False):
    if normalize:
        deletion_curve = normalize_bounds(deletion_curve)
    c0 = deletion_curve[0]
    fidelity_at_sparsity = []
    for s in range(deletion_curve.shape[0]):
        fidelity_at_sparsity.append((c0-deletion_curve[s]).sum())
    return np.array(fidelity_at_sparsity)