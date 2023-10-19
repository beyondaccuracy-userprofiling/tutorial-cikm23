import math

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing, GCNConv


class GCNIIConv(MessagePassing):
    """
    The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper
    """

    def __init__(self, channels, alpha, theta=None, layer=None, 
                 shared_weights=True, cached=False, **kwargs):
        super(GCNIIConv, self).__init__(aggr='add', **kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = math.log(theta / layer + 1)
        # self.cached = cached

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.glorot(self.weight1)
        self.glorot(self.weight2)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def forward(self, x, x_0, edge_index, edge_weight=None):
        edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                            dtype=x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)

        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)