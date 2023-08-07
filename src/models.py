import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GIN, GraphSAGE, VGAE, InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

from decoders import CosineDecoder

class LinkGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, sim='inner', sigmoid=False):
        super().__init__()

        self.sigmoid = sigmoid

        if sim=='inner':
            self.decoder = InnerProductDecoder()
        elif sim=='cosine':
            self.decoder = CosineDecoder()  

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encoder(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        
        return self.decoder(z, edge_label_index, sigmoid=self.sigmoid)

    def decode_all(self, z):
        
        return self.decoder.forward_all(z, sigmoid=self.sigmoid)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index).view(-1)
    

class LinkGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, sim='inner', sigmoid=False):
        super().__init__()

        self.sigmoid = sigmoid

        self.encoder = GIN(in_channels, hidden_channels, 2, out_channels=out_channels)  # norm="batch_norm"
        if sim=='inner':
            self.decoder = InnerProductDecoder()
        elif sim=='cosine':
            self.decoder = CosineDecoder()  

    def encode(self, x, edge_index):
        
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        
        return self.decoder(z, edge_label_index, sigmoid=self.sigmoid)

    def decode_all(self, z):
        
        return self.decoder.forward_all(z, sigmoid=self.sigmoid)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index).view(-1)
    

class LinkSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, sim='inner', sigmoid=False):
        super().__init__()

        self.sigmoid = sigmoid

        self.encoder = GraphSAGE(in_channels, hidden_channels, 2, out_channels=out_channels)
        if sim=='inner':
            self.decoder = InnerProductDecoder()
        elif sim=='cosine':
            self.decoder = CosineDecoder()  

    def encode(self, x, edge_index):
        
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        
        return self.decoder(z, edge_label_index, sigmoid=self.sigmoid)

    def decode_all(self, z):
        
        return self.decoder.forward_all(z, sigmoid=self.sigmoid)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index).view(-1)
    

class VGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, enc_in_dim, enc_hidden_dim, enc_out_dim, decoder):
        super(DeepVGAE, self).__init__(encoder=VGCNEncoder(enc_in_dim,
                                                           enc_hidden_dim,
                                                           enc_out_dim),
                                       decoder=decoder)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward(z, edge_label_index)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score