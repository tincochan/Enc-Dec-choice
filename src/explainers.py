import torch_geometric.transforms as T
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, ExplainerConfig, Explanation
from torch_geometric.nn import to_captum_input, to_captum_model
from captum.attr import IntegratedGradients, Deconvolution, GuidedBackprop, LRP
# from torch_geometric.utils import k_hop_subgraph

def gnnexplainer(
        model, 
        x, 
        edge_index, 
        edge_label_index,
        return_type='raw',
        target=None):
    
    model_config = ModelConfig(
        mode='binary_classification',
        task_level='edge',
        return_type=return_type,
    )
    
    if not target:
        explainer = Explainer(
            model=model,
            explanation_type='model',
            algorithm=GNNExplainer(epochs=200),
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
        explanation = explainer(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index
        )
    else:
        explainer = Explainer(
            model=model,
            explanation_type='phenomenon',
            algorithm=GNNExplainer(epochs=200),
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=model_config,
        )
        explanation = explainer(
            x=x,
            edge_index=edge_index,
            target=target,
            edge_label_index=edge_label_index,
        )

    return explanation


def ig(
        model,
        x, 
        edge_index, 
        edge_label_index,
        return_type='raw',
        target=None):
    
    args, additional_args = to_captum_input(
                                    x, 
                                    edge_index,
                                    'node_and_edge',
                                    edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = IntegratedGradients(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args, internal_batch_size=1)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask = ig_attr_node[0, :, :].detach(), 
        edge_mask = ig_attr_edge[0, :].detach(), 
        x=x,
        edge_index = edge_index,
        edge_label_index = edge_label_index
    )

    return explanation


def deconvolution(
        model,
        x, 
        edge_index, 
        edge_label_index,
        return_type='raw',
        target=None):
    
    args, additional_args = to_captum_input(
                                    x, 
                                    edge_index,
                                    'node_and_edge',
                                    edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = Deconvolution(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask = ig_attr_node[0, :, :].detach(), 
        edge_mask = ig_attr_edge[0, :].detach(), 
        x=x,
        edge_index = edge_index,
        edge_label_index = edge_label_index
    )

    return explanation


def backprop(
        model,
        x, 
        edge_index, 
        edge_label_index,
        return_type='raw',
        target=None):
    
    args, additional_args = to_captum_input(
                                    x, 
                                    edge_index,
                                    'node_and_edge',
                                    edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = GuidedBackprop(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask = ig_attr_node[0, :, :].detach(), 
        edge_mask = ig_attr_edge[0, :].detach(), 
        x=x,
        edge_index = edge_index,
        edge_label_index = edge_label_index
    )

    return explanation

def lrp(
        model,
        x, 
        edge_index, 
        edge_label_index,
        target=None):
    
    args, additional_args = to_captum_input(
                                    x, 
                                    edge_index,
                                    'node_and_edge',
                                    edge_label_index)

    captum_model = to_captum_model(model, mask_type='node_and_edge',
                                   output_idx=0)

    ig = LRP(captum_model)

    ig_attr_node, ig_attr_edge = ig.attribute(
        args,
        additional_forward_args=additional_args)

    # subset, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
    #     [source_node, target_node], 2, edge_index, directed=False
    # )
    explanation = Explanation(
        node_mask = ig_attr_node[0, :, :].detach(), 
        edge_mask = ig_attr_edge[0, :].detach(), 
        x=x,
        edge_index = edge_index,
        edge_label_index = edge_label_index
    )

    return explanation