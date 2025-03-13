import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from embedder import embedder
from layers import FullyConnect, Discriminator, Linear_layer, SemanticAttention
from evaluation import evaluation_metrics
VERY_SMALL_NUMBER = 1e-12
INF = 1e20

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class HGNN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        self.features = self.features.to(self.args.device)
        self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph.items()}
        model = modeler(self.args).to(self.args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        cnt_wait = 0; best = 1e9; 
        self.args.batch_size = 1
        self.g = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
                               nn.ReLU(inplace=True)).to(self.args.device)

        self.g_1 = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_equidim, bias=False),
                               nn.ReLU(inplace=True)).to(self.args.device)

        self.p_1 = nn.Sequential(nn.Linear(self.args.g_equidim, self.args.p_equidim, bias=False),
                               nn.ReLU(inplace=True)).to(self.args.device)

        self.feature_distance = self.feature_distance.to(self.args.device)


        for epoch in tqdm(range(self.args.nb_epochs)):

            model.train()
            optimizer.zero_grad()

            emb_het, emb_hom = model(self.graph, self.features, self.feature_distance)
            embs_P1 = self.g(emb_het)
            embs_P2 = self.g(emb_hom)
#######################################################################
            #The second term in Eq. (10): uniformity loss
            intra_c = (embs_P1).T @ (embs_P1).contiguous()
            intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
            loss_uni = torch.log(intra_c).mean()

            intra_c_2 = (embs_P2).T @ (embs_P2).contiguous()
            intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
            loss_uni += torch.log(intra_c_2).mean()

#######################################################################
            #The first term in Eq. (10): invariance loss
            inter_c = embs_P1.T @ embs_P2
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_inv = -torch.diagonal(inter_c).sum()

#######################################################################
            #Projection and Transformation
            embs_Q2 = self.g_1(emb_het)
            embs_Q1 = self.g_1(emb_hom)
            embs_Q1_trans = self.p_1(embs_Q1)

            # The first term in Eq. (11)
            inter_c = embs_Q1_trans.T @ embs_Q2
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_spe_inv = -torch.diagonal(inter_c).sum()

#######################################################################
            # The second term in Eq. (11)
            inter_c = embs_Q1_trans.T @ embs_Q1
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_spe_nontrival_1 = torch.diagonal(inter_c).sum()

            inter_c_1 = embs_Q1_trans.T @ embs_P2
            inter_c_1 = F.normalize(inter_c_1, p=2, dim=1)
            loss_spe_nontrival_2 = torch.diagonal(inter_c_1).sum()
########################################################################

            loss_consistency =  loss_inv + self.args.gamma * loss_uni
            loss_specificity = loss_spe_inv + self.args.eta * (loss_spe_nontrival_1 + loss_spe_nontrival_2)

            loss = loss_consistency + self.args.lambbda * loss_specificity #

            loss.backward()
            optimizer.step()
   
            train_loss = loss.item()

            if (train_loss < best):
                best = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

        model.load_state_dict(
            torch.load('saved_model/HGNNbest_{}_{}.pkl'.format(self.args.dataset, self.args.custom_key)))

        embs_het, emb_hom = model.embed(self.graph, self.features, self.feature_distance)

        h_concat = []
        h_concat.append(embs_het)
        h_concat.append(emb_hom)
        h_concat = torch.cat(h_concat, 1)
        test_out = h_concat.detach().cpu().numpy()


        if self.args.dataset in ['freebase', 'Aminer', 'imdb', 'Freebase']: #, 'ogbn-mag'
            ev = evaluation_metrics(test_out, self.labels, self.args, self.train_idx, self.val_idx, self.test_idx)
        else:
            ev = evaluation_metrics(test_out, self.labels, self.args)
        fis, fas = ev.evalutation(self.args)
        return fis, fas


class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bnn = nn.ModuleDict()
        self.disc2 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.mlp = MLP([self.args.ft_size , self.args.out_ft])
        self.semanticatt = nn.ModuleDict()
            
        for t, rels in self.args.nt_rel.items():  # {note_type: [rel1, rel2]}
            self.fc[t] = FullyConnect(args.hid_units2+args.ft_size, args.out_ft)
            self.disc2[t] = Discriminator(args.ft_size,args.out_ft)
            for rel in rels:
                self.bnn['0'+rel] = Linear_layer(args.ft_size, args.hid_units, act=nn.ReLU(), isBias=False)
                self.bnn['1'+rel] = Linear_layer(args.hid_units, args.hid_units2, act=nn.ReLU(), isBias=False)

            self.semanticatt['0'+t] = SemanticAttention(args.hid_units, args.hid_units//4)
            self.semanticatt['1'+t] = SemanticAttention(args.hid_units2, args.hid_units2//4)


    def forward(self, graph, features, distance):
        embs1 = torch.zeros((self.args.node_size, self.args.hid_units)).to(self.args.device)
        embs2 = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.args.device)

        for n, rels in self.args.nt_rel.items():   # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]
                mean_neighbor = torch.spmm(graph[n][j], features[self.args.node_cnt[t]])
                v = self.bnn['0'+rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)

            embs1[self.args.node_cnt[n]] = v_summary

        
        for n, rels in self.args.nt_rel.items():   # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                mean_neighbor = torch.spmm(graph[n][j], embs1[self.args.node_cnt[t]])
                v = self.bnn['1'+rel](mean_neighbor)  
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)
            v_cat = torch.hstack((v_summary, features[self.args.node_cnt[n]]))
            v_summary = self.fc[n](v_cat)
            
            embs2[self.args.node_cnt[n]] = v_summary
        emb_f = self.mlp(features)
        emb_f = emb_f[0:self.args.node_num]
        if self.args.dataset in ['ACM']:
            embs_het = embs1
        else:
            embs_het = embs2

        coe2 = 1.0 / self.args.beta
        res = torch.mm(torch.transpose(emb_f, 0, 1), emb_f)  # H.T* H
        inv = torch.inverse(torch.eye(emb_f.shape[1]).to(self.args.device) + coe2 * res)  # Q中的逆矩阵
        res = torch.mm(inv, res)  # B中第二项的后面一部分
        res = coe2 * emb_f - coe2 * coe2 * torch.mm(emb_f, res)  #B
        tmp = torch.mm(torch.transpose(emb_f, 0, 1), res)  # H.T * B
        part1 = torch.mm(emb_f, tmp)

        part2 = (- self.args.alpha / 2 )* torch.mm(distance, res)#/ self.args.alpha
        embs_all = part1 + part2
        embs_hom = embs_all[0:self.args.node_num]
        embs_het = embs_het[0:self.args.node_num]

        return embs_het, embs_hom

    def embed(self, graph, features, distance):
        embs1 = torch.zeros((self.args.node_size, self.args.hid_units)).to(self.args.device)
        embs2 = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.args.device)
        for n, rels in self.args.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]

                mean_neighbor = torch.spmm(graph[n][j], features[self.args.node_cnt[t]])
                v = self.bnn['0' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)
            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)

            embs1[self.args.node_cnt[n]] = v_summary


        for n, rels in self.args.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                mean_neighbor = torch.spmm(graph[n][j], embs1[self.args.node_cnt[t]])
                v = self.bnn['1' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, ft_size)
            v_summary = torch.mean(vec, 0)  # (Nt, hd)
            v_cat = torch.hstack((v_summary, features[self.args.node_cnt[n]]))
            v_summary = self.fc[n](v_cat)

            embs2[self.args.node_cnt[n]] = v_summary
        emb_f = self.mlp(features)
        emb_f = emb_f[0:self.args.node_num]
        if self.args.dataset in ['ACM']:  # and self.args.custom_key == 'Node' , 'Freebase'
            embs_het = embs1
        else:
            embs_het = embs2

        ##############################################
        coe2 = 1.0 / self.args.beta
        res = torch.mm(torch.transpose(emb_f, 0, 1), emb_f)  # H.T* H
        inv = torch.inverse(torch.eye(emb_f.shape[1]).to(self.args.device) + coe2 * res)  # Q中的逆矩阵
        res = torch.mm(inv, res)  # B中第二项的后面一部分
        res = coe2 * emb_f - coe2 * coe2 * torch.mm(emb_f, res)  #B
        tmp = torch.mm(torch.transpose(emb_f, 0, 1), res)  # H.T * B
        part1 = torch.mm(emb_f, tmp)
        part2 = (- self.args.alpha / 2) * torch.mm(distance, res)

        embs_all = part1 + part2
        embs_hom = embs_all[0:self.args.node_num]
        embs_het = embs_het[0:self.args.node_num]


        return  embs_het.detach(), embs_hom.detach()


class MLP(nn.Module):
    def __init__(self, dim,  dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = []
        for i in range(len(dim)):
            struc.append(dim[i])
        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y
