import torch
import torch.nn.functional as F
from utils import process, data_load
class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        if args.dataset == "ACM":
            features, feature_distance, node_rel, edge_dict, labels, node_idx = data_load.load_ACM(args)
        if args.dataset == "Yelp":
            features, feature_distance, node_rel, edge_dict, labels, node_idx = data_load.load_Yelp(args)
        if args.dataset == "DBLP":
            features, feature_distance, node_rel, edge_dict, labels, node_idx = data_load.load_DBLP(args)
        if args.dataset == "Aminer":
            features, feature_distance, node_rel, edge_dict, labels, self.train_idx, self.val_idx, self.test_idx, node_idx = data_load.load_Aminer(args)
        elif args.dataset in ['computers', 'photo','cs', 'physics']:
            graph, feat, labels, idx_train, idx_val, idx_test, adj_fusion, num_class = data_load.load_homogeneous_graph(args.dataset)
            args.nb_nodes = feat.shape[0]
            args.nb_classes = num_class
            args.ft_size = feat.shape[1]
            subgraph = graph.add_self_loop()
            feat1 = process.random_aug(feat, args.dfr)
            feat2 = process.random_aug(feat, args.dfr)
            features = []
            features.append(feat1)
            features.append(feat2)
            feature_distance = process.pairwise_distance(features[1])
            feature_distance = F.normalize(feature_distance)
            kthvalue = torch.kthvalue(
                feature_distance.view(feature_distance.shape[0] * feature_distance.shape[1], 1).T,
                int(feature_distance.shape[0] * feature_distance.shape[1] * args.edge_rate))[0]
            mask = (feature_distance > kthvalue).detach().float()
            feature_distance = (feature_distance * mask)
            self.train_idx = torch.tensor(idx_train.type(torch.long))
            self.val_idx = torch.tensor(idx_val.type(torch.long))
            self.test_idx = torch.tensor(idx_test.type(torch.long))
            # self.args = args

        if args.dataset in ["ACM", "Yelp", "DBLP", "Aminer"]:
            subgraph = {}
            for nt, rels in node_rel.items():
                rel_list = []
                for rel in rels:
                    s, t = rel.split('-')
                    if args.dataset == "Yelp":
                        e = edge_dict[rel][node_idx[s], :][:, node_idx[t]]
                    else:
                        e = edge_dict[rel][0][node_idx[s], :][:, node_idx[t]]
                    e = process.normalize_adj(e)
                    e = process.sparse_to_tuple(e)
                    rel_list.append(torch.sparse_coo_tensor(torch.LongTensor(e[0]), torch.FloatTensor(e[1]), torch.Size(e[2])))
                subgraph[nt] = rel_list
            args.nt_rel = node_rel
            args.node_cnt = node_idx
            args.node_type = list(args.node_cnt)
            args.ft_size = features.shape[1]
            args.node_size = features.shape[0]

        self.graph = subgraph
        self.features = features
        self.feature_distance = feature_distance
        self.labels = labels
        self.args = args

        print("Dataset: %s" % args.dataset)
        print("learning rate: %s" % args.lr)
        if args.gpu_num == "cpu":
            print("use cpu")
        else:
            print("use cuda")
