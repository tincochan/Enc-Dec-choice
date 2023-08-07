import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def train(
        model,
        optimizer,
        train_data
        ):
    model.train()
    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index,
                train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(
    model,
    data
    ):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def train_vgae(
        model,
        optimizer,
        train_data
        ):
    model.train()
    optimizer.zero_grad()

    loss = model.loss(train_data.x, train_data.edge_label_index[:, train_data.edge_label==1], train_data.edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test_vgae(
    model,
    train_data,
    test_data
    ):
    model.eval()
    roc_auc, ap = model.single_test(train_data.x,
                                    train_data.edge_label_index[:, train_data.edge_label==1],
                                    test_data.edge_label_index[:, test_data.edge_label==1],
                                    test_data.edge_label_index[:, test_data.edge_label==0])
    return roc_auc