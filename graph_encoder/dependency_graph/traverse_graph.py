import pickle
import re


def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", name.lower()))
    return any(word in words for word in test_phrases)

class RepoSearcher:
    def __init__(self, graph):
        self.G = graph

    def subgraph(self, nids):
        subg = self.G.subgraph(nids)
        edges = list(subg.edges(data='type'))
        node_data = self.get_data(nids)
        return edges, node_data

    def one_hop_neighbors(self, nid, return_data=False):
        # get one-hop neighbors from networkx graph
        if not return_data:
            return_list = []
            for nid in list(self.G.predecessors(nid)) + list(self.G.successors(nid)):
                if ':' in nid:
                    file_name = nid.split(':')[0]
                else:
                    file_name = nid
                if is_test(file_name):
                    continue
                else:
                    return_list.append(nid)
            return set(return_list)

        neigh_data = []
        for pn in self.G.predecessors(nid):
            ndata = self.get_data([pn])[0]
            for key, attr in self.G.get_edge_data(pn, nid).items():
                ndata['relation'] = attr['type'] + '-by'
                neigh_data.append(ndata)
        for sn in self.G.successors(nid):
            ndata = self.get_data([sn])[0]
            for key, attr in self.G.get_edge_data(nid, sn).items():
                ndata['relation'] = attr['type']
                neigh_data.append(ndata)

        return neigh_data

    def two_hop_neighbors(self, nid, return_data=False):
        # get two-hop neighbors from networkx graph
        one_hop = self.one_hop_neighbors(nid)
        two_hop = []
        for nid in one_hop:
            two_hop.extend(self.one_hop_neighbors(nid))
        two_hop = set(two_hop)

        return self.get_data(two_hop) if return_data else two_hop

    def dfs(self, root_nid, depth):
        # perform depth-first search on networkx graph
        visited = []
        stack = [(root_nid, 0)]
        while stack:
            nid, level = stack.pop()
            if nid not in visited:
                visited.append(nid)
                if level < depth:
                    successors = []
                    for _nid in list(self.G.successors(nid)):
                        if ':' in _nid:
                            file_name = _nid.split(':')[0]
                        else:
                            file_name = _nid
                        if is_test(file_name):
                            continue
                        else:
                            successors.append(_nid)
                    stack.extend(
                        [(n, level + 1) for n in successors]
                    )
        return visited

    def call_stack(self, root_nid, depth):
        # perform depth-first search on networkx graph
        visited = []
        stack = [(root_nid, 0)]
        while stack:
            nid, level = stack.pop()
            if nid not in visited:
                visited.append(nid)
                if level < depth:
                    successors = []
                    for nid in list(self.G.successors(nid)):
                        ndata = self.get_data([nid])[0]
                        if ndata['type'] != 'function' or is_test(ndata['file_path']):
                            continue                        
                        else:
                            successors.append(nid)
                    stack.extend(
                        [(n, level + 1) for n in successors]
                    )
        return visited
    
    def bfs(self, root_nid, depth):
        # perform breadth-first search on networkx graph
        visited = []
        queue = [(root_nid, 0)]
        while queue:
            nid, level = queue.pop(0)
            if nid not in visited:
                visited.append(nid)
                if level < depth:
                    queue.extend(
                        [(n, level + 1) for n in self.one_hop_neighbors(nid)]
                    )
        return visited

    def get_all_nodes_by_file(self, file_pattern, ntype=None):
        all_inner_nodes = []
        for node, _ntype in self.G.nodes(data='type'):
            if _ntype == 'file' and re.match(file_pattern, node):
                all_inner_nodes.extend(
                    self.get_all_inner_nodes(node, ntype)
                )
        return self.get_data(all_inner_nodes)

    def get_all_inner_nodes(self, src_node, ntype=None):
        assert ntype in ['function', 'class', None]
        inner_nodes = []
        for _, dst_node, attr in self.G.edges(src_node, data=True):
            if attr['type'] == 'contains':
                if self.G.nodes[dst_node]['type'] == ntype or ntype is None:
                    inner_nodes.append(dst_node)
                    inner_nodes.extend(self.get_all_inner_nodes(dst_node, ntype))
        return inner_nodes

    def get_data(self, nids):
        rtn = []
        for nid in nids:
            node = self.G.nodes[nid]
            path_list = nid.split(':')
            rtn.append({
                'file_path': path_list[0],
                'module_name': path_list[1] if len(path_list) > 1 else '',
                'type': node['type'],
                # 'code': node['code'],
                'start_line': node.get('start_line', 0),
                'end_line': node.get('end_line', 0)
            })
        return rtn


if __name__ == '__main__':
    repo = 'sympy__sympy-22005'
    graph = pickle.load(
        open(f"DATA/dependency_graph_v2/{repo}.pkl", "rb")
    )
    searcher = RepoSearcher(graph)
    breakpoint()