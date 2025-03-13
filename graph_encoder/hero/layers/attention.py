import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, rel_size, in_ft):
        super().__init__()
        self.w_list = nn.ModuleList([nn.Linear(in_ft, in_ft, bias=False) for _ in range(rel_size)])
        self.y_list = nn.ModuleList([nn.Linear(in_ft, 1) for _ in range(rel_size)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h 


class NodeAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_ft, out_ft, concat=True):
        super().__init__()
        self.concat = concat
        self.out_ft = out_ft
        self.W = nn.Parameter(torch.zeros(size=(in_ft, out_ft)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_ft, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, adj):
        h = torch.mm(x, self.W)  # N*d
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_ft)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        print(e.shape)

        zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
#        attention = F.dropout(attention, self.nd_dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
#            return F.elu(h_prime)
            return F.leaky_relu(h_prime)
        else:
            return h_prime


class SemanticAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.W = nn.Parameter(torch.zeros(size=(in_ft, out_ft)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_ft)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_ft)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, nb_rel):
        h = torch.mm(x, self.W)
        #h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0],1))
        #h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(nb_rel, -1)
        #semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1,keepdim=True)
        #semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
#         print(semantic_attentions)
        semantic_attentions = semantic_attentions.view(nb_rel, 1,1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_ft)
#        print(semantic_attentions)

        #input_embedding = P*N*F
        input_embedding = x.view(nb_rel,N,self.in_ft)

        #h_embedding = N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding


class LocalAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_ft, nb_rel):
        super().__init__()
        self.nb_rel = nb_rel
        self.out_ft = in_ft
        self.weight = nn.Parameter(torch.empty(in_ft, in_ft))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, v, u):
        # emb (rel_size, batch_size, d) weight (rel_size, d, 1)
        h = torch.mm(v, self.weight)  # (rel_size*node_size, d)
        h = h.view(self.nb_rel, -1, self.out_ft)
        uh = u.repeat(self.nb_rel, 1).view(self.nb_rel, -1, self.out_ft)
        a = (h * uh).sum(-1).unsqueeze(2) # (rel_size, node_size, 1)
        a = F.softmax(a, dim=0)
        local = v.view(self.nb_rel, -1, self.out_ft)# (rel_size, node_size, d)

        #h_embedding = N*F
        out = (a * local).sum(0).squeeze() + u

        return out
