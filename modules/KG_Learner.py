import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, isBias=True):
        super(GraphConvolution, self).__init__()
        self.fc_1 = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()
        self.isBias = isBias

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, input, adj):
        seq = self.fc_1(input)
        seq = torch.spmm(adj, seq)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class KGLearner(nn.Module):
    def __init__(self, embed_dim) :
        super(KGLearner, self).__init__()
        # self.gc1 = GraphConvolution(embed_dim, embed_dim)
        self.attn_fc = torch.nn.Linear(2 * embed_dim, 1, bias=False)
        self.attn_act = torch.nn.LeakyReLU(0.2)
        # self.act = torch.nn.ReLU()
        # self.evt_fc = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_fc = torch.nn.Linear(2 * embed_dim, embed_dim, bias=False)
        # self.out_fc = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(2 * embed_dim, 4 * embed_dim)),
        #     ("gelu", torch.nn.ReLU()),
        #     ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        # ]))

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, adj, subevent, event):
        num_sevt = subevent.shape[0]
        num_evt = event.shape[0]
        evt = event.float().unsqueeze(1) * adj.unsqueeze(-1).to_dense()  # (num_event, num_sub_event, dim)
        sevt = subevent.float().unsqueeze(0) * adj.unsqueeze(-1).to_dense()  # (num_event, num_sub_event, dim)
        concat_evt = torch.cat([sevt, evt], -1) # (num_event, num_sub_event, 2*dim)
        out_attn = self.attn_act(self.attn_fc(concat_evt)).squeeze(-1)  # (num_event, num_sub_event)
        attn_weights = torch.softmax(out_attn.view(num_evt, num_sevt), -1)  # (num_event, num_sub_event)
        sevt_evt = torch.sum(attn_weights.unsqueeze(-1) * sevt, 1)  # (num_event, dim)
        # new_evt = torch.cat([event.float(), sevt_evt], -1)  # (num_event, 2 * dim)
        # out = self.out_fc(new_evt)
        out = (event.float() + sevt_evt) / 2
        output = out.type(subevent.dtype)

        return output
