from fairseq.modules.layer_norm import LayerNorm
from fairseq import utils
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import (global_add_pool,
                                global_mean_pool,
                                global_max_pool)
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from molecule.features import get_bond_feature_dims, get_atom_feature_dims, \
    get_self_loops_typeid
from typing import Callable
import logging


logger = logging.getLogger(__name__)


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


class AtomHead(nn.Module):

    def __init__(self, emb_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim,)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(emb_dim)

        if weight is None:
            weight = nn.Linear(emb_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, node_feartures, masked_atom):
        if masked_atom is not None:
            node_feartures = node_feartures[masked_atom, :]
        
        x = self.dense(node_feartures)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class GINConv(MessagePassing):

    def __init__(self, embed_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(2*embed_dim, embed_dim))
        self.edge_embedding = MulOnehotEncoder(embed_dim, get_bond_feature_dims)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = edge_attr.new_zeros((x.size(0), len(get_bond_feature_dims())))
        self_loop_attr[:, 0] = get_self_loops_typeid()
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, JK='last', dropout=0., gnn_type="gin",
                 graph_pooling="mean", freeze_bn=False, activation_fn='relu'):
        super().__init__()
        self.num_layer = num_layer
        self.dropout = dropout
        self.JK = JK

        assert self.num_layer > 1

        self.atom_embedding = MulOnehotEncoder(emb_dim, get_atom_feature_dims)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr='add'))
            else:
                raise NotImplementedError()

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            bn = nn.BatchNorm1d(emb_dim)
            if freeze_bn:
                bn.momentum = 0
            self.batch_norms.append(bn)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.pool == "max":
            self.pool = global_max_pool
        elif self.pool == "attention":
            raise NotImplementedError()
        elif self.pool[:-1] == "set2set":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid graph pooling type.")

        self.atom_head = AtomHead(
            emb_dim, get_atom_feature_dims()[0], activation_fn,
            weight=self.atom_embedding.atom_embedding_list[0].weight
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.atom_embedding(x)
        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        else:
            raise NotImplementedError()

        graph_representation = self.pool(node_representation, batch=data.batch)
        if hasattr(data, 'masked_pos') and data.masked_pos is not None:
            pred_atoms = self.atom_head(node_representation, data.masked_pos)
        else:
            pred_atoms = None 
        return graph_representation, node_representation, pred_atoms

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != "" else ""
        if hasattr(self, 'atom_head'):
            cur_state = self.atom_head.state_dict()
            for k, v in cur_state.items():
                if prefix + 'atom_head.' + k not in state_dict:
                    logger.info("Overwriting " + prefix + 'atom_head.' + k)
                    state_dict[prefix + 'atom_head.' + k] = v


class GNN_graphpred(nn.Module):

    def __init__(self, num_layer, embed_dim, JK='last', dropout=0.,
                 gnn_type="gin", graph_pooling="mean"):
        super().__init__()
        self.gnn = GNN(num_layer, embed_dim, JK=JK, dropout=dropout, gnn_type=gnn_type,
                       graph_pooling=graph_pooling)



        raise NotImplementedError




