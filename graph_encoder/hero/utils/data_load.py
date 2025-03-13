import numpy as np
from collections import defaultdict
import pickle
import torch
import torch as th
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,WikiCSDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, CoraFullDataset
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
from utils import process
import os.path as osp
np.random.seed(0)


def load_ACM(args):
    path = "./dataset/" + "ACM"
    with open(path + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(path + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(path + "/node_features.pkl", "rb") as f:
        features = pickle.load(f)
    edge_dict = defaultdict(list)
    edge_dict['p-a'].append(edges[0])
    edge_dict['a-p'].append(edges[1])
    edge_dict['p-s'].append(edges[2])
    edge_dict['s-p'].append(edges[3])
    node_rel = defaultdict(list)

    node_rel['p'].append('p-a')
    node_rel['p'].append('p-s')
    node_rel['a'].append('a-p')
    node_rel['s'].append('s-p')
    node_idx = {}
    node_idx['p'] = torch.LongTensor([i for i in range(3025)])
    node_idx['a'] = torch.LongTensor([i for i in range(3025, 8937)])
    node_idx['s'] = torch.LongTensor([i for i in range(8937, 8994)])
    args.node_num = 3025
    features = process.preprocess_features(features, norm=True)
    feature_distance = process.pairwise_distance(features[0:args.node_num])
    feature_distance = F.normalize(feature_distance)
    kthvalue = torch.kthvalue(
        feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
        int(feature_distance.shape[0] * feature_distance.shape[1] * args.edge_rate))[0]
    mask = (feature_distance > kthvalue).detach().float()
    feature_distance = (feature_distance * mask)

    return features, feature_distance, node_rel, edge_dict, labels, node_idx


def load_Yelp(args):
    path = "./dataset/" + "Yelp"
    with open(path + '/meta_data.pkl', 'rb') as f:
        data = pickle.load(f)
    node_idx = {}
    for t in data['t_info'].keys():
        node_idx[t] = torch.LongTensor([i for p, i in data['node2gid'].items() if p.startswith(t)])
    with open(path + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(path + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(path + "/node_features.pkl", "rb") as f:
        features = pickle.load(f)
    node_rel = defaultdict(list)
    for rel in edges:
        s, t = rel.split('-')
        node_rel[s].append(rel)

    args.node_num = np.array(labels[0]).shape[0] + np.array(labels[1]).shape[0] + \
                    np.array(labels[2]).shape[0]
    features = process.preprocess_features(features, norm=True)
    feature_distance = process.pairwise_distance(features[0:args.node_num])

    feature_distance = F.normalize(feature_distance)
    kthvalue = torch.kthvalue(
        feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
        int(feature_distance.shape[0] * feature_distance.shape[1] * args.edge_rate))[0]
    mask = (feature_distance > kthvalue).detach().float()
    feature_distance = (feature_distance * mask)

    return features, feature_distance, node_rel, edges, labels, node_idx

def load_DBLP(args):
    path = "./dataset/" + "DBLP"
    with open(path + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(path + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(path + "/node_features.pkl", "rb") as f:
        features = pickle.load(f)
    edge_dict = defaultdict(list)

    edge_dict['p-a'].append(edges[0])
    edge_dict['a-p'].append(edges[1])
    edge_dict['p-c'].append(edges[2])
    edge_dict['c-p'].append(edges[3])
    node_rel = defaultdict(list)

    node_rel['p'].append('p-a')
    node_rel['p'].append('p-c')
    node_rel['a'].append('a-p')
    node_rel['c'].append('c-p')
    node_idx = {}
    node_idx['a'] = torch.LongTensor([i for i in range(4057)])
    node_idx['p'] = torch.LongTensor([i for i in range(4057, 18385)])
    node_idx['c'] = torch.LongTensor([i for i in range(18385, 18405)])

    args.node_num = np.array(labels[0]).shape[0] + np.array(labels[1]).shape[0] + \
                    np.array(labels[2]).shape[0]
    features = process.preprocess_features(features, norm=True)
    feature_distance = process.pairwise_distance(features[0:args.node_num])
    feature_distance = F.normalize(feature_distance)

    return features, feature_distance, node_rel, edge_dict, labels, node_idx

def load_Aminer(args):
    edge_dict = defaultdict(list)
    ratio = [20, 40, 60]
    node_types = ['paper', 'author', 'reference']
    raw_dir = "./dataset/" + "Aminer/raw/"
    features = []
    for i, node_type in enumerate(node_types):
        x = np.load(
            osp.join(raw_dir, f'features_{i}.npy'))
        features.append(torch.from_numpy(x).to(torch.float))
    features = np.array(torch.cat(features, dim=0))
    labels = np.load(osp.join(raw_dir, 'labels.npy')).astype('int32')
    labels = torch.from_numpy(labels)

    train = [np.load(raw_dir + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(raw_dir + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(raw_dir + "val_" + str(i) + ".npy") for i in ratio]

    # label = torch.FloatTensor(labels)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    pa = np.loadtxt(osp.join(raw_dir, 'pa.txt'))
    pa = torch.from_numpy(pa).t()
    pr = np.loadtxt(osp.join(raw_dir, 'pr.txt'))
    pr = torch.from_numpy(pr).t()

    edges = []
    edges.append(sp.csr_matrix((np.ones(pa[[0, 1]].long().shape[1]), (pa[[0, 1]].long()[0].numpy(),
                                                                      pa[[0, 1]].long()[1].numpy())),
                               shape=(features.shape[0], features.shape[0])))  # 'movie', 'actor'
    edges.append(sp.csr_matrix(
        (np.ones(pa[[1, 0]].long().shape[1]), (pa[[1, 0]].long()[0].numpy(), pa[[1, 0]].long()[1].numpy())),
        shape=(features.shape[0], features.shape[0])))
    edges.append(sp.csr_matrix(
        (np.ones(pr[[0, 1]].long().shape[1]), (pr[[0, 1]].long()[0].numpy(), pr[[0, 1]].long()[1].numpy())),
        shape=(features.shape[0], features.shape[0])))
    edges.append(sp.csr_matrix(
        (np.ones(pr[[1, 0]].long().shape[1]), (pr[[1, 0]].long()[0].numpy(), pr[[1, 0]].long()[1].numpy())),
        shape=(features.shape[0], features.shape[0])))

    edge_dict['p-a'].append(edges[0])
    edge_dict['a-p'].append(edges[1])
    edge_dict['p-r'].append(edges[2])
    edge_dict['r-p'].append(edges[3])
    node_rel = defaultdict(list)

    node_rel['p'].append('p-a')
    node_rel['p'].append('p-r')
    node_rel['a'].append('a-p')
    node_rel['r'].append('r-p')
    idx = {}
    idx['p'] = torch.LongTensor([i for i in range(6564)])
    idx['a'] = torch.LongTensor([i for i in range(6564, 19893)])
    idx['r'] = torch.LongTensor([i for i in range(19893, 55783)])

    features = process.preprocess_features(features, norm=True)
    args.node_num = labels.shape[0]
    feature_distance = process.pairwise_distance(features[0:args.node_num])
    feature_distance = feature_distance
    kthvalue = torch.kthvalue(
        feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
        int(feature_distance.shape[0] * feature_distance.shape[1] * args.edge_rate))[0]
    mask = (feature_distance > kthvalue).detach().float()
    feature_distance = (feature_distance * mask)
    train_idx = train[0]
    val_idx = val[0]
    test_idx = test[0]
    node_idx = idx
    return features, feature_distance, node_rel, edge_dict, labels, train_idx, val_idx, test_idx, node_idx


def load_homogeneous_graph(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'computers':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()
    elif name == 'corafull':
        dataset = CoraFullDataset()
    elif name == 'wikics':
        dataset = WikiCSDataset()
    elif name == 'ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    if name in ['ogbn-arxiv']:
        graph, labels = dataset[0]
    else:
        graph = dataset[0]
        labels = graph.ndata.pop('label')
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'computers', 'cs', 'physics', 'corafull','ogbn-arxiv']

    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num].squeeze()
        val_idx = idx[train_num:val_num].squeeze()
        test_idx = idx[val_num:].squeeze()

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    if name == "wikics":
        train_idx = th.tensor(graph.ndata.pop('train_mask'))
        val_idx = th.tensor(graph.ndata.pop('val_mask'))
        test_idx = th.tensor(graph.ndata.pop('test_mask'))

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')

    return graph, feat, labels.squeeze(), train_idx, val_idx, test_idx, graph, num_class

