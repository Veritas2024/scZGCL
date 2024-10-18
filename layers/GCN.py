import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv, HeteroConv


class GNN_Encoder(nn.Module):
    def __init__(self, in_features,dropout, layer_sizes,heads):
        super().__init__()

        
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for hidden_channels,head in zip(layer_sizes,heads):
            self.layers.append(MultiHeadAttentionLayer(in_features,hidden_channels,dropout,heads=head))
            self.batchnorms.append(BatchNorm(hidden_channels))
            in_features = hidden_channels


    def forward(self, data):
        X = data.x
        adj = data.edge_index.to_dense(dtype=torch.int64).to(torch.device('cuda:0'))
        for i, layer in enumerate(self.layers):
            X = layer(X,adj)
            X = self.batchnorms[i](X)
            X = F.relu(X)

        return X

    def reset_parameters(self):
        self.model.reset_parameters()

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features,dropout, heads, concat=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.attentions = [GraphAttentionLayer(in_features, out_features,dropout, concat=concat) for _ in range(heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([torch.unsqueeze(att(x, adj), 0) for att in self.attentions])
        x = torch.mean(x, dim=0)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'