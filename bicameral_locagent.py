import argparse
import json
import os

from query_interface import index_repo, query_repo, interactive


def _parse_args():
    parser = argparse.ArgumentParser(prog='bicameral_locagent')
    subparsers = parser.add_subparsers(dest='command', required=True)

    index_parser = subparsers.add_parser('index', help='Index a codebase')
    index_parser.add_argument('--repo', required=True, help='Path to the source repo')
    index_parser.add_argument('--output', required=True, help='Output directory for indexes')

    query_parser = subparsers.add_parser('query', help='Query an indexed codebase')
    query_parser.add_argument('--index', required=True, help='Path to index directory')
    query_parser.add_argument('--query', required=True, help='Natural language query')
    query_parser.add_argument('--output', default='', help='Output JSON file')
    query_parser.add_argument('--config', default='', help='Path to config.yaml')

    interactive_parser = subparsers.add_parser('interactive', help='Interactive query mode')
    interactive_parser.add_argument('--index', required=True, help='Path to index directory')
    interactive_parser.add_argument('--config', default='', help='Path to config.yaml')

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.command == 'index':
        meta = index_repo(args.repo, args.output)
        print(json.dumps(meta, indent=2))
        return

    if args.command == 'query':
        output_path = args.output or None
        result = query_repo(args.index, args.query, output_path=output_path, config_path=args.config or None)
        if not output_path:
            print(json.dumps(result, indent=2))
        return

    if args.command == 'interactive':
        interactive(args.index, config_path=args.config or None)
        return


if __name__ == '__main__':
    main()
