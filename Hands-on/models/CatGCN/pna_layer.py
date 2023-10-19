from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing, GCNConv
# from torch_geometric.nn.conv.gcn_conv import gcn_norm

# implementation of 'GCNConv.norm' method of Pytorch Geometric v1.3.2 (not present in the latest version)
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

def gcn_norm_old(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class PNAConv(MessagePassing):
    """
    Pure neighborhood aggregation layer.
    """
    def __init__(self, K=1, cached=False, bias=True, **kwargs):
        super(PNAConv, self).__init__(aggr='add', **kwargs)
        
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm_old(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
