import argparse
import json
import os
from collections import Counter

from dependency_graph.build_graph import build_graph


def summarize_graph(graph):
    node_types = Counter([data.get('type') for _, data in graph.nodes(data=True)])
    edge_types = Counter([data.get('type') for _, _, data in graph.edges(data=True)])
    return {
        'nodes': dict(node_types),
        'edges': dict(edge_types),
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    graph = build_graph(args.repo, global_import=True)
    summary = summarize_graph(graph)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
