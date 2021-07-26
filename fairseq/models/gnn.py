from typing import Optional, Callable
import torch
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch import nn, Tensor
from molecule.features import get_atom_feature_dims, get_bond_feature_dims
from fairseq import utils
from torch_geometric.nn import global_max_pool, global_mean_pool, global_sort_pool


class CustomMessagePassing(MessagePassing):
    def __init__(self, aggr: Optional[str] = "maxminmean", embed_dim: Optional[int] = None):
        if aggr in ['maxminmean']:
            super().__init__(aggr=None)
            self.aggr = aggr
            assert embed_dim is not None
            self.aggrmlp = nn.Linear(3 * embed_dim, embed_dim)
        else:
            super().__init__(aggr=aggr)

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor],
                  dim_size: Optional[int]) -> Tensor:
        if self.aggr in ['maxminmean']:
            inputs_fp32 = inputs.float()
            input_max = scatter(inputs_fp32,
                                index,
                                dim=self.node_dim,
                                dim_size=dim_size,
                                reduce='max')
            input_min = scatter(inputs_fp32,
                                index,
                                dim=self.node_dim,
                                dim_size=dim_size,
                                reduce='min')
            input_mean = scatter(inputs_fp32,
                                 index,
                                 dim=self.node_dim,
                                 dim_size=dim_size,
                                 reduce='mean')
            aggr_out = torch.cat([input_max, input_min, input_mean], dim=-1).type_as(inputs)
            aggr_out = self.aggrmlp(aggr_out)
            return aggr_out
        else:
            return super().aggregate(inputs, index, ptr, dim_size)


class MulOnehotEncoder(nn.Module):
    def __init__(self, embed_dim, get_feature_dims: Callable):
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()

        for dim in get_feature_dims():
            emb = nn.Embedding(dim, embed_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i])
        return x_embedding


class ResidualConvLayer(CustomMessagePassing):
    def __init__(self,
                 in_dim,
                 emb_dim,
                 aggr,
                 encode_edge=False,
                 bond_encoder=False,
                 edge_feat_dim=None):
        super().__init__(aggr, embed_dim=in_dim)
        self.mlp = nn.Linear(in_dim, emb_dim)
        self.encode_edge = encode_edge

        if encode_edge:
            if bond_encoder:
                self.edge_encoder = MulOnehotEncoder(in_dim, get_bond_feature_dims)
            else:
                self.edge_encoder = nn.Linear(edge_feat_dim, in_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.encode_edge and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None
        m = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        h = x + m
        out = self.mlp(h)
        return out

    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            msg = x_j + edge_attr
        else:
            msg = x_j

        return msg


def get_norm_layer(norm, fea_dim):
    norm = norm.lower()

    if norm == 'layer':
        return nn.LayerNorm(fea_dim)
    elif norm == "batch":
        return nn.BatchNorm1d(fea_dim)
    else:
        raise NotImplementedError()


class AtomHead(nn.Module):
    def __init__(self, emb_dim, output_dim, activation_fn, weight=None, norm=None):
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.norm = get_norm_layer(norm, emb_dim)

        if weight is None:
            weight = nn.Linear(emb_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, node_features, masked_atom):
        if masked_atom is not None:
            node_features = node_features[masked_atom, :]

        x = self.dense(node_features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class DeeperGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.gnn_number_layer
        self.dropout = args.gnn_dropout
        self.conv_encode_edge = args.conv_encode_edge
        self.embed_dim = args.gnn_embed_dim
        self.aggr = args.gnn_aggr
        self.norm = args.gnn_norm

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation_fn = utils.get_activation_fn(getattr(args, 'gnn_activation_fn', 'relu'))

        for layer in range(self.num_layers):
            self.gcns.append(
                ResidualConvLayer(
                    self.embed_dim,
                    self.embed_dim,
                    self.aggr,
                    encode_edge=self.conv_encode_edge,
                    bond_encoder=True,
                ))
            self.norms.append(get_norm_layer(self.norm, self.embed_dim))

        self.atom_encoder = MulOnehotEncoder(self.embed_dim, get_atom_feature_dims)
        if not self.conv_encode_edge:
            self.bond_encoder = MulOnehotEncoder(self.embed_dim, get_bond_feature_dims)

        self.graph_pred_linear = nn.Identity()
        self.output_features = 2 * self.embed_dim
        self.atom_head = AtomHead(self.embed_dim,
                                  get_atom_feature_dims()[0],
                                  getattr(args, 'gnn_activation_fn', 'relu'),
                                  norm=self.norm,
                                  weight=self.atom_encoder.atom_embedding_list[0].weight)

    def forward(self, graph, masked_tokens=None, features_only=False):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        batch = graph.batch

        h = self.atom_encoder(x)

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        h = self.gcns[0](h, edge_index, edge_emb)

        for layer in range(1, self.num_layers):
            residual = h
            h = self.norms[layer](h)
            h = self.activation_fn(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcns[layer](h, edge_index, edge_emb)
            h = h + residual
        h = self.norms[0](h)
        h = self.activation_fn(h)
        node_fea = F.dropout(h, p=self.dropout, training=self.training)

        graph_fea = self.pool(node_fea, batch)

        if not features_only:
            atom_pred = self.atom_head(node_fea, masked_tokens)
        else:
            atom_pred = None

        return (graph_fea, node_fea), atom_pred

    def pool(self, h, batch):
        h_fp32 = h.float()
        h_max = global_max_pool(h_fp32, batch)
        h_mean = global_mean_pool(h_fp32, batch)
        h = torch.cat([h_max, h_mean], dim=-1).type_as(h)
        h = self.graph_pred_linear(h)
        return h
