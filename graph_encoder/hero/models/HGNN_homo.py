import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tqdm import tqdm
from embedder import embedder
from layers import *
from utils.process import GCN
from evaluation import evaluation_metrics

class HGNN_homo(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        self.features = [features.to(self.args.device) for features in self.features]
        self.graph  = self.graph.to(self.args.device)
        model = modeler(self.args).to(self.args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
        cnt_wait = 0
        best = 1e9
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
            embs_ori, emb_hom = model(self.graph, self.features, self.feature_distance)

            embs_P1 = self.g(embs_ori)
            embs_P2 = self.g(emb_hom)

#################################################################################
            #The second term in Eq. (10): uniformity loss
            intra_c = (embs_P1).T @ (embs_P1).contiguous()
            intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
            loss_uni = torch.log(intra_c).mean()

            intra_c_2 = (embs_P2).T @ (embs_P2).contiguous()
            intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
            loss_uni += torch.log(intra_c_2).mean()
#################################################################################
            #The first term in Eq. (10): invariance loss
            inter_c = embs_P1.T @ embs_P2
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_inv = -torch.diagonal(inter_c).sum()

#######################################################################
            # Projection and Transformation
            embs_Q2 = self.g_1(embs_ori)
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

            loss_consistency = loss_inv + self.args.gamma * loss_uni
            loss_specificity = loss_spe_inv + self.args.eta * (loss_spe_nontrival_1 + loss_spe_nontrival_2)

            loss = loss_consistency + self.args.lambbda * loss_specificity  #
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

        embs_ori, emb_hom = model.embed(self.graph, self.features, self.feature_distance)

        h_concat = []
        h_concat.append(embs_ori)
        h_concat.append(emb_hom)
        h_concat = torch.cat(h_concat, 1)
        test_out = h_concat.detach().cpu().numpy()
        if self.args.dataset in ['photo', 'computers', 'cs', 'physics']:
            ev = evaluation_metrics(test_out, self.labels, self.args, self.train_idx, self.val_idx, self.test_idx)
        else:
            ev = evaluation_metrics(test_out, self.labels, self.args)
        fis, fas = ev.evalutation(args=self.args)
        return fis, fas


class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gcn = GCN(self.args.ft_size, self.args.hid_units, self.args.out_ft, 2, self.args)
        self.mlp = MLP([self.args.ft_size , self.args.out_ft]) #self.args.out_ft, + self.args.struc_size


    def forward(self, graph, features,distance):
        embs_ori = self.gcn(graph, features[0])
        emb_f = self.mlp(features[1])
        coe2 = 1.0 / self.args.beta
        res = torch.mm(torch.transpose(emb_f, 0, 1), emb_f)  # H.T* H
        inv = torch.inverse(torch.eye(emb_f.shape[1]).to(self.args.device) + coe2 * res)  # Q中的逆矩阵
        res = torch.mm(inv, res)  # B中第二项的后面一部分
        res = coe2 * emb_f - coe2 * coe2 * torch.mm(emb_f, res)  #B
        tmp = torch.mm(torch.transpose(emb_f, 0, 1), res)  # H.T * B
        part1 = torch.mm(emb_f, tmp)
        part2 = (- self.args.alpha / 2) * torch.mm(distance, res)
        embs_hom = part1 + part2

        return embs_ori, embs_hom

    def embed(self, graph, features,distance):
        embs_ori = self.gcn(graph, features[0])
        emb_f = self.mlp(features[1])
        coe2 = 1.0 / self.args.beta
        res = torch.mm(torch.transpose(emb_f, 0, 1), emb_f)  # H.T* H
        inv = torch.inverse(torch.eye(emb_f.shape[1]).to(self.args.device) + coe2 * res)  # Q中的逆矩阵
        res = torch.mm(inv, res)  # B中第二项的后面一部分
        res = coe2 * emb_f - coe2 * coe2 * torch.mm(emb_f, res)  # B
        tmp = torch.mm(torch.transpose(emb_f, 0, 1), res)  # H.T * B
        part1 = torch.mm(emb_f, tmp)
        part2 = (- self.args.alpha / 2) * torch.mm(distance, res)
        embs_hom = part1 + part2
        return embs_ori.detach(), embs_hom.detach()
    


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
        y = self.net[-1](x)

        return y
