import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

try:
    from tree_sitter_languages import get_language, get_parser
except Exception:  # pragma: no cover - handled at runtime
    get_language = None
    get_parser = None

VERSION = 'v3.0'
NODE_TYPE_DIRECTORY = 'directory'
NODE_TYPE_FILE = 'file'
NODE_TYPE_CLASS = 'class'
NODE_TYPE_FUNCTION = 'function'
EDGE_TYPE_CONTAINS = 'contains'
EDGE_TYPE_INHERITS = 'inherits'
EDGE_TYPE_INVOKES = 'invokes'
EDGE_TYPE_IMPORTS = 'imports'

VALID_NODE_TYPES = [NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]
VALID_EDGE_TYPES = [EDGE_TYPE_CONTAINS, EDGE_TYPE_INHERITS, EDGE_TYPE_INVOKES, EDGE_TYPE_IMPORTS]

SKIP_DIRS = {'.git', '.github', 'node_modules', 'dist', 'build', '.next'}

EXTENSION_LANGUAGE = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'jsx',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.cs': 'c_sharp',
}

QUERY_FILE_MAP = {
    'jsx': 'javascript',
    'tsx': 'typescript',
}

LANGUAGE_FALLBACK = {
    'jsx': 'javascript',
    'tsx': 'typescript',
}

PARSER_CACHE: Dict[str, object] = {}
LANGUAGE_CACHE: Dict[str, object] = {}


@dataclass
class Definition:
    node_id: str
    name: str
    type: str
    code: str
    start_line: int
    end_line: int
    parent_id: str
    kind: Optional[str] = None


def _ensure_tree_sitter():
    if get_language is None or get_parser is None:
        raise ImportError(
            'tree_sitter_languages is required. Add tree-sitter and tree-sitter-languages to requirements.'
        )


def _posix_path(path: Path) -> str:
    return path.as_posix()


def _is_skip_dir(path: str) -> bool:
    parts = path.replace('\\', '/').split('/')
    return any(part in SKIP_DIRS for part in parts)


def _iter_source_files(repo_path: str) -> Iterable[Tuple[str, str, str]]:
    for root, dirs, files in os.walk(repo_path):
        rel_dir = Path(root).relative_to(repo_path).as_posix()
        if rel_dir == '.':
            rel_dir = ''
        dirs[:] = sorted([d for d in dirs if not _is_skip_dir(f"{rel_dir}/{d}".strip('/'))])
        for filename in sorted(files):
            ext = Path(filename).suffix.lower()
            language = EXTENSION_LANGUAGE.get(ext)
            if not language:
                continue
            abs_path = os.path.join(root, filename)
            if os.path.islink(abs_path):
                continue
            rel_path = Path(abs_path).relative_to(repo_path).as_posix()
            yield rel_path, abs_path, language


def _read_file_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def _get_language(language_id: str):
    _ensure_tree_sitter()
    resolved = LANGUAGE_FALLBACK.get(language_id, language_id)
    if resolved not in LANGUAGE_CACHE:
        LANGUAGE_CACHE[resolved] = get_language(resolved)
    return LANGUAGE_CACHE[resolved]


def _get_parser(language_id: str):
    _ensure_tree_sitter()
    resolved = LANGUAGE_FALLBACK.get(language_id, language_id)
    if resolved not in PARSER_CACHE:
        PARSER_CACHE[resolved] = get_parser(resolved)
    return PARSER_CACHE[resolved]


def _load_query(language_id: str):
    query_lang = QUERY_FILE_MAP.get(language_id, language_id)
    queries_dir = Path(__file__).parent / 'language_parsers'
    query_path = queries_dir / f'{query_lang}.scm'
    if not query_path.exists():
        return None
    try:
        language = _get_language(language_id)
        return language.query(query_path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _node_text(code: bytes, node) -> str:
    return code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        return text[1:-1]
    return text


def _collect_identifiers(node, code: bytes) -> List[str]:
    identifiers: List[str] = []

    def walk(child):
        if child.type in (
            'identifier',
            'type_identifier',
            'property_identifier',
            'field_identifier',
            'scoped_identifier',
            'qualified_identifier',
            'namespace_identifier',
        ):
            identifiers.append(_node_text(code, child))
        for grandchild in child.children:
            walk(grandchild)

    walk(node)
    return identifiers


def _get_name_from_node(node, code: bytes) -> Optional[str]:
    name_node = node.child_by_field_name('name')
    if name_node is None:
        return None
    return _node_text(code, name_node)


def _definition_from_node(rel_path: str, node, code: bytes, node_type: str, parent_id: str, name: str, kind: Optional[str]):
    return Definition(
        node_id=f"{rel_path}:{name}",
        name=name,
        type=node_type,
        code=_node_text(code, node),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        parent_id=parent_id,
        kind=kind,
    )


def _extract_python_defs(tree, code: bytes, rel_path: str):
    definitions: List[Definition] = []
    bases: Dict[str, List[str]] = {}

    def walk(node, class_stack: List[str]):
        if node.type == 'class_definition':
            name = _get_name_from_node(node, code)
            if not name:
                return
            full_name = '.'.join(class_stack + [name]) if class_stack else name
            parent_id = f"{rel_path}:{'.'.join(class_stack)}" if class_stack else rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_CLASS, parent_id, full_name, 'class'))
            bases[definitions[-1].node_id] = _extract_python_bases(node, code)
            class_stack.append(name)
            for child in node.children:
                walk(child, class_stack)
            class_stack.pop()
            return

        if node.type in ('function_definition', 'async_function_definition'):
            name = _get_name_from_node(node, code)
            if not name:
                return
            if class_stack:
                class_name = '.'.join(class_stack)
                full_name = f"{class_name}.{name}"
                parent_id = f"{rel_path}:{class_name}"
            else:
                full_name = name
                parent_id = rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, full_name, None))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(tree.root_node, [])
    return definitions, bases


def _extract_python_bases(class_node, code: bytes) -> List[str]:
    for child in class_node.children:
        if child.type == 'argument_list':
            return _collect_identifiers(child, code)
    return []


def _extract_js_ts_defs(tree, code: bytes, rel_path: str, language_id: str):
    definitions: List[Definition] = []
    bases: Dict[str, List[str]] = {}

    class_types = {'class_declaration'}
    if language_id in ('typescript', 'tsx'):
        class_types.update({'interface_declaration', 'type_alias_declaration', 'enum_declaration'})

    def walk(node, class_stack: List[str]):
        if node.type in class_types:
            name = _get_name_from_node(node, code)
            if not name:
                return
            full_name = '.'.join(class_stack + [name]) if class_stack else name
            parent_id = f"{rel_path}:{'.'.join(class_stack)}" if class_stack else rel_path
            kind = node.type.replace('_declaration', '')
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_CLASS, parent_id, full_name, kind))
            bases[definitions[-1].node_id] = _extract_js_ts_bases(node, code)
            class_stack.append(name)
            for child in node.children:
                walk(child, class_stack)
            class_stack.pop()
            return

        if node.type == 'method_definition':
            if not class_stack:
                return
            name = _get_name_from_node(node, code)
            if not name:
                return
            class_name = '.'.join(class_stack)
            full_name = f"{class_name}.{name}"
            parent_id = f"{rel_path}:{class_name}"
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, full_name, None))
            return

        if node.type == 'function_declaration':
            if class_stack:
                return
            name = _get_name_from_node(node, code)
            if not name:
                return
            parent_id = rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, name, None))
            return

        if node.type == 'variable_declarator':
            if class_stack:
                return
            value_node = node.child_by_field_name('value')
            if value_node is None or value_node.type not in ('arrow_function', 'function'):
                return
            name_node = node.child_by_field_name('name')
            if name_node is None:
                return
            name = _node_text(code, name_node)
            parent_id = rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, name, None))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(tree.root_node, [])
    return definitions, bases


def _extract_js_ts_bases(class_node, code: bytes) -> List[str]:
    superclass = class_node.child_by_field_name('superclass')
    if superclass is not None:
        return _collect_identifiers(superclass, code)
    for child in class_node.children:
        if child.type in ('class_heritage', 'extends_clause', 'implements_clause', 'extends_type_clause'):
            return _collect_identifiers(child, code)
    return []


def _extract_java_defs(tree, code: bytes, rel_path: str):
    definitions: List[Definition] = []
    bases: Dict[str, List[str]] = {}

    class_types = {'class_declaration', 'interface_declaration', 'enum_declaration'}

    def walk(node, class_stack: List[str]):
        if node.type in class_types:
            name = _get_name_from_node(node, code)
            if not name:
                return
            full_name = '.'.join(class_stack + [name]) if class_stack else name
            parent_id = f"{rel_path}:{'.'.join(class_stack)}" if class_stack else rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_CLASS, parent_id, full_name, node.type))
            bases[definitions[-1].node_id] = _collect_identifiers(node, code)
            class_stack.append(name)
            for child in node.children:
                walk(child, class_stack)
            class_stack.pop()
            return

        if node.type in ('method_declaration', 'constructor_declaration'):
            if not class_stack:
                return
            name = _get_name_from_node(node, code)
            if not name:
                return
            class_name = '.'.join(class_stack)
            full_name = f"{class_name}.{name}"
            parent_id = f"{rel_path}:{class_name}"
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, full_name, None))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(tree.root_node, [])
    return definitions, bases


def _extract_go_defs(tree, code: bytes, rel_path: str):
    definitions: List[Definition] = []
    bases: Dict[str, List[str]] = {}

    def walk(node, class_stack: List[str]):
        if node.type == 'type_spec':
            type_node = node.child_by_field_name('type')
            if type_node is not None and type_node.type in ('struct_type', 'interface_type'):
                name_node = node.child_by_field_name('name')
                if name_node is None:
                    return
                name = _node_text(code, name_node)
                full_name = '.'.join(class_stack + [name]) if class_stack else name
                parent_id = f"{rel_path}:{'.'.join(class_stack)}" if class_stack else rel_path
                definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_CLASS, parent_id, full_name, type_node.type))
                bases[definitions[-1].node_id] = []
                return

        if node.type in ('function_declaration', 'method_declaration'):
            name = _get_name_from_node(node, code)
            if not name:
                return
            if class_stack:
                class_name = '.'.join(class_stack)
                full_name = f"{class_name}.{name}"
                parent_id = f"{rel_path}:{class_name}"
            else:
                full_name = name
                parent_id = rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, full_name, None))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(tree.root_node, [])
    return definitions, bases


def _extract_rust_defs(tree, code: bytes, rel_path: str):
    definitions: List[Definition] = []
    bases: Dict[str, List[str]] = {}

    class_types = {'struct_item', 'enum_item', 'trait_item'}

    def walk(node, class_stack: List[str]):
        if node.type in class_types:
            name = _get_name_from_node(node, code)
            if not name:
                return
            full_name = '.'.join(class_stack + [name]) if class_stack else name
            parent_id = f"{rel_path}:{'.'.join(class_stack)}" if class_stack else rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_CLASS, parent_id, full_name, node.type))
            bases[definitions[-1].node_id] = []
            return

        if node.type == 'function_item':
            name = _get_name_from_node(node, code)
            if not name:
                return
            parent_id = rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, name, None))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(tree.root_node, [])
    return definitions, bases


def _extract_csharp_defs(tree, code: bytes, rel_path: str):
    definitions: List[Definition] = []
    bases: Dict[str, List[str]] = {}

    class_types = {'class_declaration', 'interface_declaration', 'struct_declaration', 'enum_declaration'}

    def walk(node, class_stack: List[str]):
        if node.type in class_types:
            name = _get_name_from_node(node, code)
            if not name:
                return
            full_name = '.'.join(class_stack + [name]) if class_stack else name
            parent_id = f"{rel_path}:{'.'.join(class_stack)}" if class_stack else rel_path
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_CLASS, parent_id, full_name, node.type))
            bases[definitions[-1].node_id] = _collect_identifiers(node, code)
            class_stack.append(name)
            for child in node.children:
                walk(child, class_stack)
            class_stack.pop()
            return

        if node.type in ('method_declaration', 'constructor_declaration'):
            if not class_stack:
                return
            name = _get_name_from_node(node, code)
            if not name:
                return
            class_name = '.'.join(class_stack)
            full_name = f"{class_name}.{name}"
            parent_id = f"{rel_path}:{class_name}"
            definitions.append(_definition_from_node(rel_path, node, code, NODE_TYPE_FUNCTION, parent_id, full_name, None))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(tree.root_node, [])
    return definitions, bases


def _extract_definitions(language_id: str, tree, code: bytes, rel_path: str):
    if language_id == 'python':
        return _extract_python_defs(tree, code, rel_path)
    if language_id in ('javascript', 'jsx', 'typescript', 'tsx'):
        return _extract_js_ts_defs(tree, code, rel_path, language_id)
    if language_id == 'java':
        return _extract_java_defs(tree, code, rel_path)
    if language_id == 'go':
        return _extract_go_defs(tree, code, rel_path)
    if language_id == 'rust':
        return _extract_rust_defs(tree, code, rel_path)
    if language_id == 'c_sharp':
        return _extract_csharp_defs(tree, code, rel_path)
    return [], {}


def _extract_imports(language_id: str, tree, code: bytes) -> List[str]:
    imports: List[str] = []

    def walk(node):
        if language_id == 'python':
            if node.type in ('import_statement', 'import_from_statement'):
                imports.extend(_parse_python_import_statement(_node_text(code, node)))
                return
        if language_id in ('javascript', 'jsx', 'typescript', 'tsx'):
            if node.type in ('import_statement', 'export_statement'):
                source_node = node.child_by_field_name('source')
                if source_node is not None:
                    imports.append(_strip_quotes(_node_text(code, source_node)))
                return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imports


def _parse_python_import_statement(text: str) -> List[str]:
    text = text.strip().replace('\n', ' ')
    if text.startswith('import '):
        content = text[len('import '):]
        parts = [p.strip() for p in content.split(',') if p.strip()]
        modules = []
        for part in parts:
            if ' as ' in part:
                name, _alias = part.split(' as ', 1)
                modules.append(name.strip())
            else:
                modules.append(part)
        return modules

    if text.startswith('from '):
        try:
            rest = text[len('from '):]
            module, _import_part = rest.split(' import ', 1)
            return [module.strip()]
        except ValueError:
            return []

    return []


def _resolve_python_module(module: str, file_dir: str, repo_path: str) -> Optional[str]:
    if module is None:
        return None
    module = module.strip()
    if not module:
        return None

    leading = len(module) - len(module.lstrip('.'))
    module_rest = module.lstrip('.')

    if leading > 0:
        base_dir = PurePosixPath(file_dir)
        for _ in range(leading):
            base_dir = base_dir.parent
        if module_rest:
            rel_base = base_dir / PurePosixPath(module_rest.replace('.', '/'))
        else:
            rel_base = base_dir
        candidate = PurePosixPath(rel_base)
    else:
        candidate = PurePosixPath(module.replace('.', '/'))

    for suffix in ('.py', ''):
        cand = PurePosixPath(str(candidate) + suffix)
        path = Path(repo_path) / cand
        if path.is_file():
            return path.relative_to(repo_path).as_posix()

    init_path = Path(repo_path) / candidate / '__init__.py'
    if init_path.is_file():
        return init_path.relative_to(repo_path).as_posix()

    return None


def _resolve_ts_module(module: str, file_dir: str, repo_path: str) -> Optional[str]:
    if module is None:
        return None
    module = module.strip()
    if not module:
        return None
    if not (module.startswith('.') or module.startswith('/')):
        return None

    if module.startswith('/'):
        module_path = os.path.normpath(module.lstrip('/'))
    else:
        module_path = os.path.normpath(f\"{file_dir}/{module}\")
    module_path = module_path.replace('\\\\', '/')

    candidates = []
    for ext in ('.ts', '.tsx', '.js', '.jsx', '.d.ts'):
        candidates.append(Path(repo_path) / f"{module_path}{ext}")
    candidates.append(Path(repo_path) / module_path / 'index.ts')
    candidates.append(Path(repo_path) / module_path / 'index.tsx')
    candidates.append(Path(repo_path) / module_path / 'index.js')
    candidates.append(Path(repo_path) / module_path / 'index.jsx')

    for cand in candidates:
        if cand.is_file():
            return cand.relative_to(repo_path).as_posix()

    return None


def _collect_call_names(language_id: str, code: str) -> List[str]:
    if not code.strip():
        return []
    try:
        parser = _get_parser(language_id)
    except Exception:
        return []
    tree = parser.parse(bytes(code, 'utf-8'))
    call_names: List[str] = []

    if language_id == 'python':
        call_node_types = {'call'}
    elif language_id in ('javascript', 'jsx', 'typescript', 'tsx'):
        call_node_types = {'call_expression', 'new_expression'}
    elif language_id == 'java':
        call_node_types = {'method_invocation', 'object_creation_expression'}
    elif language_id == 'go':
        call_node_types = {'call_expression'}
    elif language_id == 'rust':
        call_node_types = {'call_expression'}
    elif language_id == 'c_sharp':
        call_node_types = {'invocation_expression', 'object_creation_expression'}
    else:
        call_node_types = set()

    def walk(node):
        if node.type in call_node_types:
            name = _extract_call_name(language_id, node, bytes(code, 'utf-8'))
            if name:
                call_names.append(name)
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return call_names


def _extract_call_name(language_id: str, node, code: bytes) -> Optional[str]:
    if language_id == 'python':
        func_node = node.child_by_field_name('function') or (node.named_children[0] if node.named_children else None)
        if func_node is None:
            return None
        if func_node.type == 'identifier':
            return _node_text(code, func_node)
        if func_node.type == 'attribute':
            attr = func_node.child_by_field_name('attribute')
            if attr:
                return _node_text(code, attr)
        return _extract_last_identifier(func_node, code)

    if language_id in ('javascript', 'jsx', 'typescript', 'tsx'):
        func_node = node.child_by_field_name('function') or (node.named_children[0] if node.named_children else None)
        if func_node is None:
            return None
        if func_node.type == 'identifier':
            return _node_text(code, func_node)
        if func_node.type in ('member_expression', 'optional_member_expression'):
            prop = func_node.child_by_field_name('property')
            if prop:
                return _node_text(code, prop)
        return _extract_last_identifier(func_node, code)

    if language_id == 'java':
        if node.type == 'method_invocation':
            name_node = node.child_by_field_name('name')
            if name_node:
                return _node_text(code, name_node)
        if node.type == 'object_creation_expression':
            type_node = node.child_by_field_name('type')
            if type_node:
                return _extract_last_identifier(type_node, code)

    if language_id == 'go':
        name = node.child_by_field_name('function')
        if name:
            return _extract_last_identifier(name, code)

    if language_id == 'rust':
        func = node.child_by_field_name('function') or (node.named_children[0] if node.named_children else None)
        if func:
            return _extract_last_identifier(func, code)

    if language_id == 'c_sharp':
        if node.type == 'invocation_expression':
            expr = node.child_by_field_name('expression')
            if expr:
                return _extract_last_identifier(expr, code)
        if node.type == 'object_creation_expression':
            type_node = node.child_by_field_name('type')
            if type_node:
                return _extract_last_identifier(type_node, code)

    return _extract_last_identifier(node, code)


def _extract_last_identifier(node, code: bytes) -> Optional[str]:
    last = None
    cursor = node.walk()
    reached = False
    while True:
        if cursor.node.type in (
            'identifier',
            'type_identifier',
            'property_identifier',
            'field_identifier',
        ):
            last = _node_text(code, cursor.node)
        if not cursor.goto_first_child():
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    reached = True
                    break
            if reached:
                break
    return last


def _ensure_dir_nodes(graph: nx.MultiDiGraph, rel_path: str):
    rel_dir = PurePosixPath(rel_path).parent
    if str(rel_dir) in ('.', ''):
        return
    parts = rel_dir.parts
    cur = '/'
    for part in parts:
        nxt = PurePosixPath(cur) / part if cur != '/' else PurePosixPath(part)
        nxt_str = nxt.as_posix()
        if not graph.has_node(nxt_str):
            graph.add_node(nxt_str, type=NODE_TYPE_DIRECTORY)
        if not graph.has_edge(cur, nxt_str):
            graph.add_edge(cur, nxt_str, type=EDGE_TYPE_CONTAINS)
        cur = nxt_str


def _build_file_candidate_map(graph: nx.MultiDiGraph) -> Dict[str, Dict[str, List[str]]]:
    file_candidates: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    file_nodes = [nid for nid, data in graph.nodes(data=True) if data.get('type') == NODE_TYPE_FILE]

    for file_id in file_nodes:
        for node_id, data in graph.nodes(data=True):
            if data.get('type') not in (NODE_TYPE_CLASS, NODE_TYPE_FUNCTION):
                continue
            if not node_id.startswith(f"{file_id}:"):
                continue
            short = node_id.split(':', 1)[1]
            short_tail = short.split('.')[-1]
            file_candidates[file_id][short].append(node_id)
            if short_tail != short:
                file_candidates[file_id][short_tail].append(node_id)

    import_adj: Dict[str, List[str]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        if data.get('type') != EDGE_TYPE_IMPORTS:
            continue
        if graph.nodes[u].get('type') != NODE_TYPE_FILE:
            continue
        import_adj[u].append(v)

    for file_id in file_nodes:
        visited = set()
        stack = list(import_adj.get(file_id, []))
        while stack:
            imp_file = stack.pop()
            if imp_file in visited:
                continue
            visited.add(imp_file)
            if imp_file in file_candidates:
                for name, nodes in file_candidates[imp_file].items():
                    file_candidates[file_id][name].extend(nodes)
            for nxt in import_adj.get(imp_file, []):
                if nxt not in visited:
                    stack.append(nxt)

    return file_candidates


def build_graph(repo_path: str, fuzzy_search: bool = True, global_import: bool = False):
    graph = nx.MultiDiGraph()
    graph.add_node('/', type=NODE_TYPE_DIRECTORY)

    file_info: Dict[str, Dict] = {}

    for rel_path, abs_path, language_id in _iter_source_files(repo_path):
        _ensure_dir_nodes(graph, rel_path)
        graph.add_node(rel_path, type=NODE_TYPE_FILE, code=_read_file_text(abs_path))
        graph.add_edge(PurePosixPath(rel_path).parent.as_posix() if PurePosixPath(rel_path).parent.as_posix() not in ('.', '') else '/',
                       rel_path, type=EDGE_TYPE_CONTAINS)
        file_info[rel_path] = {
            'abs_path': abs_path,
            'language_id': language_id,
        }

    for rel_path, info in file_info.items():
        code_text = graph.nodes[rel_path].get('code', '')
        code_bytes = bytes(code_text, 'utf-8')
        try:
            parser = _get_parser(info['language_id'])
        except Exception:
            continue
        tree = parser.parse(code_bytes)
        definitions, bases = _extract_definitions(info['language_id'], tree, code_bytes, rel_path)
        for definition in definitions:
            graph.add_node(
                definition.node_id,
                type=definition.type,
                code=definition.code,
                start_line=definition.start_line,
                end_line=definition.end_line,
            )
            graph.add_edge(definition.parent_id, definition.node_id, type=EDGE_TYPE_CONTAINS)
        imports = _extract_imports(info['language_id'], tree, code_bytes)
        info['imports'] = imports
        info['bases'] = bases
        info['definitions'] = definitions

    for rel_path, info in file_info.items():
        file_dir = PurePosixPath(rel_path).parent.as_posix()
        language_id = info['language_id']
        for module in info.get('imports', []):
            if language_id == 'python':
                resolved = _resolve_python_module(module, file_dir, repo_path)
            elif language_id in ('javascript', 'jsx', 'typescript', 'tsx'):
                resolved = _resolve_ts_module(module, file_dir, repo_path)
            else:
                resolved = None
            if resolved and graph.has_node(resolved):
                graph.add_edge(rel_path, resolved, type=EDGE_TYPE_IMPORTS, alias=None)

    file_candidates = _build_file_candidate_map(graph)

    for rel_path, info in file_info.items():
        language_id = info['language_id']
        candidate_map = file_candidates.get(rel_path, {})
        for definition in info.get('definitions', []):
            if definition.type not in (NODE_TYPE_CLASS, NODE_TYPE_FUNCTION):
                continue
            call_names = _collect_call_names(language_id, definition.code)
            for name in set(call_names):
                for target in candidate_map.get(name, []):
                    if target != definition.node_id:
                        graph.add_edge(definition.node_id, target, type=EDGE_TYPE_INVOKES)

            if definition.type == NODE_TYPE_CLASS:
                base_names = info.get('bases', {}).get(definition.node_id, [])
                for base in set(base_names):
                    for target in candidate_map.get(base, []):
                        if target != definition.node_id:
                            graph.add_edge(definition.node_id, target, type=EDGE_TYPE_INHERITS)

    return graph


def write_graph_outputs(graph: nx.MultiDiGraph, output_path: str, tags_path: Optional[str] = None):
    import json
    import pickle

    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)

    if not tags_path:
        return

    def _file_end_line(code: str) -> int:
        return max(1, len(code.splitlines())) if code is not None else 1

    with open(tags_path, 'w', encoding='utf-8') as f:
        for node_id, data in graph.nodes(data=True):
            record = {
                'node_id': node_id,
                'type': data.get('type'),
                'start_line': data.get('start_line', 1),
                'end_line': data.get('end_line', _file_end_line(data.get('code', ''))),
                'code': data.get('code', ''),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_bm25_index(repo_path: str, output_dir: str):
    import json
    import pickle

    try:
        import bm25s
    except Exception as exc:  # pragma: no cover - dependency guarded
        raise ImportError('bm25s is required to build the BM25 index') from exc

    documents: List[str] = []
    doc_ids: List[str] = []
    for rel_path, abs_path, _language_id in _iter_source_files(repo_path):
        documents.append(_read_file_text(abs_path))
        doc_ids.append(rel_path)

    tokens = bm25s.tokenize(documents, stopwords='en', show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokens, show_progress=False)

    os.makedirs(output_dir, exist_ok=True)
    index_path = Path(output_dir) / 'bm25_index.pkl'
    with open(index_path, 'wb') as f:
        pickle.dump({'bm25': bm25, 'doc_ids': doc_ids}, f)

    meta_path = Path(output_dir) / 'bm25_docs.jsonl'
    with open(meta_path, 'w', encoding='utf-8') as f:
        for doc_id in doc_ids:
            f.write(json.dumps({'doc_id': doc_id}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='graph.pkl')
    parser.add_argument('--tags_output', type=str, default='')
    parser.add_argument('--bm25_output', type=str, default='')
    args = parser.parse_args()

    graph = build_graph(args.repo_path, global_import=True)
    write_graph_outputs(graph, args.output, args.tags_output or None)

    if args.bm25_output:
        build_bm25_index(args.repo_path, args.bm25_output)


if __name__ == '__main__':
    main()
