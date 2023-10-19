import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class BatchAGC(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BatchAGC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        support = torch.bmm(x, expand_weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BatchFiGNN(nn.Module):
    def __init__(self, f_in, f_out, out_channels):
        super(BatchFiGNN, self).__init__()
        # Edge Weights
        self.a_src = Parameter(torch.Tensor(f_in, 1))
        self.a_dst = Parameter(torch.Tensor(f_in, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        # Transformation
        self.w = Parameter(torch.Tensor(f_in, f_out))
        self.bias = Parameter(torch.Tensor(f_out))
        # State Update by GRU
        self.rnn = torch.nn.GRUCell(f_out, f_out, bias=True)
        # Attention Pooling
        self.mlp_1 = nn.Linear(f_out, out_channels, bias=True)
        self.mlp_2 = nn.Linear(f_out, 1, bias=True)

        init.xavier_uniform_(self.w)
        init.constant_(self.bias, 0)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj, steps):
        bs, n = h.size()[:2]
        ## Edge Weights  
        attn_src = torch.matmul(h, self.a_src)
        attn_dst = torch.matmul(h, self.a_dst)
        attn = attn_src.expand(-1, -1, n) + \
            attn_dst.expand(-1, -1, n).permute(0, 2, 1)
        attn = self.leaky_relu(attn)
        mask = torch.eye(adj.size()[-1]).unsqueeze(0).cuda()
        mask = 1 - mask
        attn = attn * mask
        attn = self.softmax(attn)
        ## State Update
        s = h
        for _ in range(steps):
            ## Transformation
            a = torch.matmul(s, self.w)
            a = torch.matmul(attn, a) + self.bias
            ## GRU
            s = self.rnn(s.view(-1, s.size()[-1]), a.view(-1, a.size()[-1]))
            s = h.view(h.size()) + h
        ## Attention Pooling
        output = self.mlp_1(s)
        weight = self.mlp_2(s).permute(0, 2, 1)
        output = torch.matmul(weight, output).squeeze()
        return output

class BatchGAT(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchGAT, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + \
            attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
