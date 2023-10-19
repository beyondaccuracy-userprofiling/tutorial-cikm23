import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP

from gnn_layers import BatchAGC, BatchFiGNN, BatchGAT
from pna_layer import PNAConv
from gcnii_layer import GCNIIConv

class StackedGNN(nn.Module):
    """
    Multi-layer GNN model.
    """
    def __init__(self, args, field_count, field_size, output_channels):
        """
        :param args: Arguments object.
        :param field_count: Number of fields.
        :param field_size: Number of sampled fields for each user.
        :param output_channels: Number of target classes.
        """
        super(StackedGNN, self).__init__()
        self.args = args

        if self.args.grn_units != 'none':
            self.grn_units = [args.field_dim] + [int(x) for x in args.grn_units.strip().split(",")] + [output_channels]
        else:
            self.grn_units = [args.field_dim] + [output_channels]
        if self.args.nfm_units != 'none':
            self.nfm_units = [args.field_dim] + [int(x) for x in args.nfm_units.strip().split(",")] + [output_channels]
        else:
            self.nfm_units = [args.field_dim] + [output_channels]

        self.input_channels = args.field_dim
        self.output_channels = output_channels

        # For Baseline
        if self.args.gnn_units != 'none':
            self.gnn_units = [self.input_channels] + [int(x) for x in args.gnn_units.strip().split(",")] + [self.output_channels]
        else:
            self.gnn_units = [self.input_channels] + [self.output_channels]

        self.field_count = field_count
        self.field_size = field_size


        self.field_embedding = nn.Embedding(field_count, args.field_dim)
        self.field_embedding.weight.requires_grad = True
                
        self._setup_layers()

    def _setup_layers(self):
        """
        Creating the layers based on the args.
        """
        # Categorical feature interaction modeling
        ''' Global interaction modeling '''
        if self.args.graph_refining == 'agc':
            self.grn = BatchAGC(self.args.field_dim, self.args.field_dim)
            self.num_grn_layer = len(self.grn_units) - 1
            self.grn_layer_stack = nn.ModuleList()
            for i in range(self.num_grn_layer):
                self.grn_layer_stack.append(
                        nn.Linear(self.grn_units[i], self.grn_units[i + 1], bias=True))
        elif self.args.graph_refining == 'gat':
            n_heads = [int(x) for x in self.args.multi_heads.strip().split(",")]
            attn_dropout = 0. 
            # attn_dropout = self.args.dropout
            self.gat_units = [int(x) for x in self.args.gat_units.strip().split(",")]
            self.num_gat_layer = len(self.gat_units) - 1
            self.gat_layer_stack = nn.ModuleList()
            for i in range(self.num_gat_layer):
                f_in = self.gat_units[i] * n_heads[i - 1] if i else self.gat_units[i] 
                self.gat_layer_stack.append(
                        BatchGAT(
                            n_heads[i], f_in=f_in,
                            f_out=self.gat_units[i + 1], attn_dropout=attn_dropout))
            self.num_grn_layer = len(self.grn_units) - 1
            self.grn_layer_stack = nn.ModuleList()
            for i in range(self.num_grn_layer):
                self.grn_layer_stack.append(
                        nn.Linear(self.grn_units[i], self.grn_units[i + 1], bias=True))
        elif self.args.graph_refining == 'cosimi':
            self.num_grn_layer = len(self.grn_units) - 1
            self.grn_layer_stack = nn.ModuleList()
            for i in range(self.num_grn_layer):
                self.grn_layer_stack.append(
                        nn.Linear(self.grn_units[i], self.grn_units[i + 1], bias=True))

        ''' Local interaction modeling '''
        if self.args.bi_interaction == 'nfm':
            self.num_nfm_layer = len(self.nfm_units) - 1
            self.nfm_layer_stack = nn.ModuleList()
            for i in range(self.num_nfm_layer):
                self.nfm_layer_stack.append(
                        nn.Linear(self.nfm_units[i], self.nfm_units[i + 1], bias=True))

        # GNN Layer
        if self.args.graph_layer == 'gcn':
            self.gnn_layers = nn.ModuleList()
            for i, _ in enumerate(self.gnn_units[:-1]):
                self.gnn_layers.append(GCNConv(self.gnn_units[i], self.gnn_units[i+1]))
        elif self.args.graph_layer == 'gat_1':
            self.gnn_layers = GATConv(self.input_channels, self.output_channels, heads=1, concat=True, negative_slope=0.2, dropout=self.args.dropout, bias=True)
        elif self.args.graph_layer == 'gat_2':
            n_heads = 8
            self.gnn_layers_1 = GATConv(self.input_channels, self.gnn_units[1], heads=n_heads, concat=True, negative_slope=0.2, dropout=0, bias=True)
            self.gnn_layers_2 = GATConv(self.gnn_units[1]*n_heads, self.output_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True)
        elif self.args.graph_layer == 'sgc':
            self.gnn_layers = SGConv(self.input_channels, self.output_channels, K=self.args.gnn_hops, cached=False)
        elif self.args.graph_layer == 'appnp':
            self.num_mlp_layer = len(self.gnn_units) - 1
            self.mlp_layer_stack = nn.ModuleList()
            for i in range(self.num_mlp_layer):
                self.mlp_layer_stack.append(
                        nn.Linear(self.gnn_units[i], self.gnn_units[i + 1], bias=True))
            self.gnn_layers = APPNP(K=10, alpha=0.1, bias=True)
        elif self.args.graph_layer == 'cat-appnp':
            self.gnn_layers = APPNP(K=10, alpha=0.1, bias=True)
        elif self.args.graph_layer == 'gcnii_F':
            self.num_gnn_layer = self.args.gnn_hops
            self.lin_layer_1 = nn.Linear(self.input_channels, self.gnn_units[1], bias=True)
            self.gnn_layers = nn.ModuleList()
            for layer in range(self.num_gnn_layer):
                self.gnn_layers.append(GCNIIConv(self.gnn_units[1], alpha=self.args.alpha, theta=self.args.theta, layer=layer+1, shared_weights=False))
            self.lin_layer_2 = nn.Linear(self.gnn_units[1], self.output_channels, bias=True)
        elif self.args.graph_layer == 'gcnii_T':
            self.num_gnn_layer = self.args.gnn_hops
            self.lin_layer_1 = nn.Linear(self.input_channels, self.gnn_units[1], bias=True)
            self.gnn_layers = nn.ModuleList()
            for layer in range(self.num_gnn_layer):
                self.gnn_layers.append(GCNIIConv(self.gnn_units[1], alpha=self.args.alpha, theta=self.args.theta, layer=layer+1, shared_weights=True))
            self.lin_layer_2 = nn.Linear(self.gnn_units[1], self.output_channels, bias=True)
        elif self.args.graph_layer == 'cross_1':
            self.mlp_layers_1 = nn.Linear(self.input_channels, self.output_channels, bias=False)
            self.mlp_layers_2 = nn.Linear(self.input_channels, self.output_channels, bias=False)
            self.gnn_layers = PNAConv(K=1, cached=False)
        elif self.args.graph_layer == 'cross_2':
            self.mlp_layers_11 = nn.Linear(self.input_channels, self.gnn_units[1], bias=False)
            self.mlp_layers_12 = nn.Linear(self.input_channels, self.gnn_units[1], bias=False)
            self.gnn_layers_1 = PNAConv(K=1, cached=False)
            self.mlp_layers_21 = nn.Linear(self.gnn_units[1], self.output_channels, bias=False)
            self.mlp_layers_22 = nn.Linear(self.gnn_units[1], self.output_channels, bias=False)
            self.gnn_layers_2 = PNAConv(K=1, cached=False)
        elif self.args.graph_layer == 'fignn':
            self.fi_layers = BatchFiGNN(self.input_channels, self.gnn_units[1], self.output_channels)
            self.gnn_layers = PNAConv(K=self.args.gnn_hops, cached=False)
        elif self.args.graph_layer == 'pna':
            self.gnn_layers = PNAConv(K=self.args.gnn_hops, cached=False)

    def forward(self, edges, field_index, field_adjs):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :parm field_index: User-field index matrix.
        :parm field_adjs: Normalized adjacency matrix with probe coefficient.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        raw_field_feature = self.field_embedding(field_index)

        # Categorical feature interaction modeling
        ''' Global interaction modeling '''
        field_feature = raw_field_feature
        
        if self.args.graph_refining == 'agc':
            field_feature = self.grn(field_feature, field_adjs.float())
            field_feature = F.relu(field_feature)
            field_feature = F.dropout(field_feature, self.args.dropout, training=self.training)
            
            if self.args.aggr_pooling == 'mean':
                user_feature = torch.mean(field_feature, dim=-2)

            for i, grn_layer in enumerate(self.grn_layer_stack):
                user_feature = grn_layer(user_feature)
                if i + 1 < self.num_grn_layer:
                    user_feature = F.relu(user_feature)
                    user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            user_gnn_feature = user_feature
        
        elif self.args.graph_refining == 'gat':
            bs, n = field_adjs.size()[:2]
            for i, gat_layer in enumerate(self.gat_layer_stack):
                field_feature = gat_layer(field_feature, field_adjs.byte()) 
                if i + 1 == self.num_gat_layer:
                    field_feature = field_feature.mean(dim=1) 
                else:
                    field_feature = F.elu(field_feature.transpose(1, 2).contiguous().view(bs, n, -1)) 
                    field_feature = F.dropout(field_feature, self.args.dropout, training=self.training)

            if self.args.aggr_pooling == 'mean':
                user_feature = torch.mean(field_feature, dim=-2)
            for i, grn_layer in enumerate(self.grn_layer_stack):
                user_feature = grn_layer(user_feature)
                if i + 1 < self.num_grn_layer:
                    user_feature = F.relu(user_feature)
                    user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            user_gnn_feature = user_feature
        
        elif self.args.graph_refining == 'cosimi':
            similarity_mat = torch.bmm(field_feature, field_feature.permute(0, 2, 1)) 
            feature_norm = torch.sqrt(torch.sum(torch.mul(field_feature, field_feature), dim=-1)).unsqueeze(2) 
            cosine_distance = torch.div(similarity_mat, torch.mul(feature_norm, feature_norm.permute(0, 2, 1))) 
            field_feature = torch.bmm(cosine_distance, field_feature) 

            if self.args.aggr_pooling == 'mean':
                user_feature = torch.mean(field_feature, dim=-2)

            for i, grn_layer in enumerate(self.grn_layer_stack):
                user_feature = grn_layer(user_feature)
                if i + 1 < self.num_grn_layer:
                    user_feature = F.relu(user_feature)
                    user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            user_gnn_feature = user_feature

        ''' Local interaction modeling '''
        field_feature = raw_field_feature
        if self.args.bi_interaction == 'nfm': 
            # sum-square-part
            summed_field_feature = torch.sum(field_feature, 1) 
            square_summed_field_feature = summed_field_feature ** 2 
            # squre-sum-part
            squared_field_feature = field_feature ** 2 
            sum_squared_field_feature = torch.sum(squared_field_feature, 1) 
            # second order
            user_feature = 0.5 * (square_summed_field_feature - sum_squared_field_feature)
            # deep part
            for i, nfm_layer in enumerate(self.nfm_layer_stack):
                user_feature = nfm_layer(user_feature)
                if i + 1 < self.num_nfm_layer:
                    user_feature = F.relu(user_feature)
                    user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            user_nfm_feature = user_feature

        # Aggregation
        if self.args.aggr_style == 'sum':
            user_feature = self.args.balance_ratio*user_gnn_feature + \
                (1-self.args.balance_ratio)*user_nfm_feature 

        if self.args.graph_refining == 'none' and self.args.bi_interaction == 'none':
            user_feature = torch.mean(raw_field_feature, dim=-2)

        # GNN Layer
        if self.args.graph_layer == 'gcn':
            for i, _ in enumerate(self.gnn_units[:-2]): 
                user_feature = F.relu(self.gnn_layers[i](user_feature, edges))
                if i > 1:
                    user_feature = F.dropout(user_feature, p=self.args.dropout, training=self.training)
            user_feature = self.gnn_layers[i+1](user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'gat_1': 
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'gat_2': 
            user_feature = F.elu(self.gnn_layers_1(user_feature, edges))
            user_feature = F.dropout(user_feature, p=self.args.dropout, training=self.training)
            user_feature = self.gnn_layers_2(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'sgc':
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'appnp':
            for i, mlp_layer in enumerate(self.mlp_layer_stack):
                user_feature = mlp_layer(user_feature)
                if i + 1 < self.num_mlp_layer:
                    user_feature = F.relu(user_feature)
                    user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'cat-appnp':
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1) 
        elif self.args.graph_layer == 'gcnii_F' or self.args.graph_layer == 'gcnii_T':
            user_feature = self.lin_layer_1(user_feature)
            user_feature = F.relu(user_feature)
            user_feature = user_feature_0 = F.dropout(user_feature, self.args.dropout, training=self.training)
            for i, gnn_layer in enumerate(self.gnn_layers):
                    user_feature = gnn_layer(user_feature, user_feature_0, edges)
                    if i + 1 < self.num_gnn_layer:
                        user_feature = F.relu(user_feature)
                        user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            user_feature = self.lin_layer_2(user_feature)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'cross_1':
            alpha = 1
            user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            x_1 = self.mlp_layers_1(user_feature)
            x_2 = self.mlp_layers_1(user_feature)
            x_sec_ord = torch.mul(x_1, x_2) * alpha 
            user_feature = x_1 + x_2 + x_sec_ord
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'cross_2':
            alpha = 1
            user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            x_11 = self.mlp_layers_11(user_feature)
            x_12 = self.mlp_layers_12(user_feature)
            x_sec_ord_1 = torch.mul(x_11, x_12) * alpha 
            user_feature = x_11 + x_12 + x_sec_ord_1
            user_feature = self.gnn_layers_1(user_feature, edges)
            user_feature = F.dropout(user_feature, self.args.dropout, training=self.training)
            x_21 = self.mlp_layers_21(user_feature)
            x_22 = self.mlp_layers_22(user_feature)
            x_sec_ord_2 = torch.mul(x_21, x_22) * alpha 
            user_feature = x_21 + x_22 + x_sec_ord_2
            user_feature = self.gnn_layers_2(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'fignn':
            user_feature = self.fi_layers(raw_field_feature, field_adjs.float(), self.args.num_steps)
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'pna':
            user_feature = self.gnn_layers(user_feature, edges)
            predictions = F.log_softmax(user_feature, dim=1)
        elif self.args.graph_layer == 'none':
            predictions = F.log_softmax(user_feature, dim=1)
        return predictions
