import argparse
import ast
import os
import re
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt


def handle_edge_cases(code):
    # hard-coded edge cases
    code = code.replace('\ufeff', '')
    code = code.replace('constants.False', '_False')
    code = code.replace('constants.True', '_True')
    code = code.replace("False", "_False")
    code = code.replace("True", "_True")
    code = code.replace("DOMAIN\\username", "DOMAIN\\\\username")
    code = code.replace("Error, ", "Error as ")
    code = code.replace('Exception, ', 'Exception as ')
    code = code.replace("print ", "yield ")
    pattern = r'except\s+\(([^,]+)\s+as\s+([^)]+)\):'
    # Replace 'as' with ','
    code = re.sub(pattern, r'except (\1, \2):', code)
    code = code.replace("raise AttributeError as aname", "raise AttributeError")
    return code


def find_imports(filepath, repo_path):
    # root_path: 项目根目录
    try:
        with open(filepath, 'r') as file:
            tree = ast.parse(file.read(), filename=filepath)
    except:
        raise SyntaxError

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # 绝对导入:
                import_entities = []
                # print(node.module)
                # imports.append(node.module)
                for alias in node.names:
                    import_entities.append(alias.name)
                imports.append({"module": node.module, "entities": import_entities})
            else:  # 相对导入
                # 计算相对路径
                import_entities = []
                relative_module_parts = filepath.replace(repo_path, '').strip(os.sep).split(os.sep)[
                                        :-node.level]
                relative_module = '.'.join(relative_module_parts)
                if node.module:
                    relative_module += '.' + node.module
                for alias in node.names:
                    import_entities.append(alias.name)
                imports.append({"module": relative_module, "entities": import_entities})
    return imports


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.classes = []
        self.class_codes = {}
        self.top_level_functions = []
        self.function_codes = {}
        self.current_class = None

    def visit_ClassDef(self, node):
        class_name = node.name
        self.classes.append(class_name)
        self.class_codes[class_name] = self._get_source_segment(node)
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        if self.current_class is None:
            function_name = node.name
            self.top_level_functions.append(function_name)
            self.function_codes[function_name] = self._get_source_segment(node)
        self.generic_visit(node)

    def _get_source_segment(self, node):
        with open(self.filename, 'r') as file:
            source_code = file.read()
        return ast.get_source_segment(source_code, node)


# 解析指定文件，使用CodeAnalyzer分析文件中的类和顶级函数
def analyze_file(filepath):
    with open(filepath, 'r') as file:
        code = file.read()
        # code = handle_edge_cases(code)
        try:
            tree = ast.parse(code, filename=filepath)
        except:
            raise SyntaxError
    analyzer = CodeAnalyzer(filepath)
    try:
        analyzer.visit(tree)
    except RecursionError:
        pass
    return (analyzer.classes, analyzer.class_codes,
            analyzer.top_level_functions, analyzer.function_codes)


# 遍历repo_path下的所有Python文件，构建文件、类和函数的依赖关系图
def build_graph(repo_path):
    graph = nx.DiGraph()
    file_nodes = {}

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                try:
                    file_path = os.path.join(root, file)
                    filename = os.path.relpath(file_path, repo_path)
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                    graph.add_node(filename, type='file', code=file_content)

                    # 为节点添加属性
                    file_nodes[filename] = file_path

                    classes, class_codes, functions, function_codes = analyze_file(file_path)
                except (UnicodeDecodeError, SyntaxError):
                    # Skip the file that cannot decode or parse
                    continue

                for cls in classes:
                    class_node = f'{filename}:{cls}'
                    class_code = class_codes[cls]
                    graph.add_node(class_node, type='class', code=class_code)
                    # print(class_code)
                    graph.add_edge(filename, class_node, type='contains')

                for func in functions:
                    func_node = f'{filename}:{func}'
                    func_code = function_codes[func]
                    graph.add_node(func_node, type='function', code=func_code)
                    # print(func_code)
                    graph.add_edge(filename, func_node, type='contains')

    for filename, filepath in file_nodes.items():
        try:
            imports = find_imports(filepath, repo_path)
        except SyntaxError:
            continue

        for imp in imports:
            imp_path = os.path.join(repo_path, imp['module'].replace('.', '/') + '.py')
            if os.path.isfile(imp_path):
                imp_filename = os.path.relpath(imp_path, repo_path)
                for entity in imp['entities']:
                    node = f'{imp_filename}:{entity}'
                    if graph.has_node(node):
                        graph.add_edge(filename, node, type='imports')
            elif os.path.isdir(os.path.join(repo_path, imp['module'])):
                is_init = False
                for entity in imp['entities']:
                    imp_path = os.path.join(repo_path, imp['module'], entity + '.py')
                    if os.path.isfile(imp_path):
                        is_init = True
                        imp_filename = os.path.relpath(imp_path, repo_path)
                        if graph.has_node(imp_filename):
                            graph.add_edge(filename, imp_filename, type='imports')
                if not is_init:
                    init_path = os.path.join(repo_path, imp['module'], '__init__.py')
                    if os.path.isfile(init_path):
                        imp_filename = os.path.relpath(init_path, repo_path)
                        for entity in imp['entities']:
                            node = f'{imp_filename}:{entity}'
                            if graph.has_node(node):
                                graph.add_edge(filename, node, type='imports')
            else:
                init_path = os.path.join(repo_path, imp['module'], '__init__.py')
                if os.path.isfile(init_path):
                    imp_filename = os.path.relpath(init_path, repo_path)
                    for entity in imp['entities']:
                        node = f'{imp_filename}:{entity}'
                        if graph.has_node(node):
                            graph.add_edge(filename, node, type='imports')

    return graph


"""
analyze_for_class_node:
    类节点为caller，找到其包含的类和函数，并建立相应的边。
    使用了analyze_compositions，analyze_helpers，analyze_decorators。
analyze_compositions：类->类
analyze_helpers：类->函数
analyze_decorators：类->函数
"""
def analyze_for_class_node(node, graph):
    class_nodes = []
    func_nodes = []

    # 找连接到此节点，并且 edge['type'] == 'contains' 的 file node
    for predecessor in graph.predecessors(node):
        if graph.edges[predecessor, node]['type'] == 'contains':
            file_node = predecessor
            # 遍历 file node 的 imports 和 contains 的节点，分别记录在 class_nodes 和 func_nodes 数组里
            for edge in graph.edges(file_node):
                if graph.edges[edge]['type'] == 'imports':
                    imported_node = edge[1]

                    if graph.nodes[imported_node]['type'] == 'class':
                        class_nodes.append(imported_node)
                    elif graph.nodes[imported_node]['type'] == 'function':
                        func_nodes.append(imported_node)
                elif graph.edges[edge]['type'] == 'contains':
                    contained_node = edge[1]

                    if graph.nodes[contained_node]['type'] == 'class' and contained_node != node:
                        class_nodes.append(contained_node)
                    elif graph.nodes[contained_node]['type'] == 'function':
                        func_nodes.append(contained_node)

    # 对 class_nodes的候选callee节点内遍历检查，连接类与类的 composition 关系
    analyze_compositions(node, graph, class_nodes)

    # 对 func_nodes 的候选callee节点内遍历检查，连接类与函数的 helper，decorator 关系
    analyze_helpers(node, graph, func_nodes)
    analyze_decorators(node, graph, func_nodes)


# 组合：类->类
def analyze_compositions(node, graph, class_nodes):
    class_source_code = graph.nodes[node]['code']

    # 解析类的源代码
    tree = ast.parse(class_source_code)

    # 获取当前类的名称
    caller_class_name = node.split(':')[-1]

    # 获取所有可能的被调用类的名称
    callee_class_names = {callee_node.split(':')[-1] for callee_node in class_nodes}

    # 存储找到的组合关系
    compositions = []

    # 遍历 AST 节点以找到组合关系
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_class_name:
            for body_item in ast_node.body:
                if isinstance(body_item, ast.FunctionDef) and body_item.name == '__init__':
                    for stmt in body_item.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(stmt.value, ast.Call):
                                    if isinstance(stmt.value.func, ast.Name):
                                        composed_class = stmt.value.func.id
                                        if composed_class in callee_class_names:
                                            compositions.append(composed_class)

    # 在图中添加组合关系的边
    for composed_class in compositions:
        composed_node = next(node for node in class_nodes if node.endswith(f":{composed_class}"))
        graph.add_edge(node, composed_node, type='composes')


# helper：类->函数
def analyze_helpers(node, graph, func_nodes):
    function_source_code = graph.nodes[node]['code']

    # 解析类的源代码
    tree = ast.parse(function_source_code)

    # 获取当前类的名称
    caller_class_name = node.split(':')[-1]

    # 获取所有可能的被调用函数的名称
    callee_function_names = {func_node.split(':')[-1] for func_node in func_nodes}

    # 存储找到的辅助函数关系
    helpers = []

    # 遍历 AST 节点以找到辅助函数关系
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_class_name:
            for body_item in ast_node.body:
                if isinstance(body_item, ast.FunctionDef):
                    for stmt in body_item.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            if isinstance(stmt.value.func, ast.Name):
                                called_function = stmt.value.func.id
                                if called_function in callee_function_names:
                                    helpers.append(called_function)
                        if isinstance(stmt, ast.Assign):
                            if isinstance(stmt.value, ast.Call):
                                if isinstance(stmt.value.func, ast.Name):
                                    called_function = stmt.value.func.id
                                    if called_function in callee_function_names:
                                        helpers.append(called_function)
                        if isinstance(stmt, ast.Return):
                            if isinstance(stmt.value, ast.Call):
                                if isinstance(stmt.value.func, ast.Name):
                                    called_function = stmt.value.func.id
                                    if called_function in callee_function_names:
                                        helpers.append(called_function)

    # 在图中添加辅助函数关系的边
    for helper in helpers:
        helper_node = next(func_node for func_node in func_nodes if func_node.endswith(f":{helper}"))
        graph.add_edge(node, helper_node, type='uses helper')

    # decorator：类->函数


def analyze_decorators(node, graph, func_nodes):
    function_source_code = graph.nodes[node]['code']

    # 解析类的源代码
    tree = ast.parse(function_source_code)

    # 获取当前类的名称
    caller_class_name = node.split(':')[-1]

    # 获取所有可能的被调用函数的名称
    decorator_function_names = {func_node.split(':')[-1] for func_node in func_nodes}

    # 存储找到的装饰器关系
    decorators = []

    # 遍历 AST 节点以找到装饰器关系
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_class_name:
            for body_item in ast_node.body:
                if isinstance(body_item, ast.FunctionDef):
                    for decorator in body_item.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorator_name = decorator.id
                            if decorator_name in decorator_function_names:
                                decorators.append(decorator_name)

    # 在图中添加装饰器关系的边
    for decorator in decorators:
        decorator_node = next(func_node for func_node in func_nodes if func_node.endswith(f":{decorator}"))
        graph.add_edge(node, decorator_node, type='uses decorator')


"""
analyze_for_function_node:
    函数节点为caller，找到其包含的类和函数，并建立相应的边。
    使用了analyze_constructors，analyze_invokes。
analyze_constructors：非成员函数->类
analyze_invokes：函数->函数
"""
def analyze_for_function_node(node, graph):
    class_nodes = []
    func_nodes = []
    # 找连接到此节点，并且 edge['type'] == 'contains' 的 file node
    for predecessor in graph.predecessors(node):
        if graph.edges[predecessor, node]['type'] == 'contains' and graph.nodes[predecessor][
            'type'] == 'file':
            file_node = predecessor
            # 遍历 file node 的 imports 和 contains 的节点，分别记录在 class_nodes 和 func_nodes 数组里
            for edge in graph.edges(file_node):
                if graph.edges[edge]['type'] == 'imports':
                    imported_node = edge[1]

                    if graph.nodes[imported_node]['type'] == 'class':
                        class_nodes.append(imported_node)
                    elif graph.nodes[imported_node]['type'] == 'function':
                        func_nodes.append(imported_node)
                elif graph.edges[edge]['type'] == 'contains':
                    contained_node = edge[1]

                    if graph.nodes[contained_node]['type'] == 'class':
                        class_nodes.append(contained_node)
                    elif graph.nodes[contained_node]['type'] == 'function' and contained_node != node:
                        func_nodes.append(contained_node)

    # 对 class_nodes的候选callee节点内遍历检查，连接非成员函数与类的 constructors 关系
    analyze_constructors(node, graph, class_nodes)

    # 对 func_nodes 的候选callee节点内遍历检查，连接函数与函数的 invokes 关系
    analyze_invokes(node, graph, func_nodes)


# 构造器：非成员函数->类
def analyze_constructors(node, graph, class_nodes):
    function_source_code = graph.nodes[node]['code']

    # 解析函数的源代码
    tree = ast.parse(function_source_code)

    # 获取当前函数的名称
    caller_function_name = node.split(':')[-1]

    # 获取所有可能的被调用类的名称
    class_names = {class_node.split(':')[-1] for class_node in class_nodes}

    # 存储找到的构造器关系
    constructors = []

    # 遍历 AST 节点以找到构造器关系
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.FunctionDef) and ast_node.name == caller_function_name:
            for stmt in ast_node.body:
                if isinstance(stmt, ast.Return):
                    if isinstance(stmt.value, ast.Call):
                        if isinstance(stmt.value.func, ast.Name):
                            called_class = stmt.value.func.id
                            if called_class in class_names:
                                constructors.append(called_class)
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, ast.Call):
                        if isinstance(stmt.value.func, ast.Name):
                            called_class = stmt.value.func.id
                            if called_class in class_names:
                                constructors.append(called_class)
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    if isinstance(stmt.value.func, ast.Name):
                        called_class = stmt.value.func.id
                        if called_class in class_names:
                            constructors.append(called_class)

    # 检查 constructors 中的每个构造器是否属于某个类的成员函数
    valid_constructors = []
    for constructor in constructors:
        constructor_node = next(
            class_node for class_node in class_nodes if class_node.endswith(f":{constructor}"))
        class_source_code = graph.nodes[constructor_node]['code']
        # 检查类的代码是否包含函数的代码
        if function_source_code not in class_source_code:
            valid_constructors.append(constructor_node)

    # 在图中添加构造器关系的边
    for constructor_node in valid_constructors:
        graph.add_edge(node, constructor_node, type='uses constructor')


# 函数->函数
def analyze_invokes(node, graph, func_nodes):
    function_source_code = graph.nodes[node]['code']

    # 解析函数的源代码
    tree = ast.parse(function_source_code)

    # 获取当前函数的名称
    caller_function_name = node.split(':')[-1]

    # 获取所有可能的被调用函数的名称
    callee_function_names = {func_node.split(':')[-1] for func_node in func_nodes}

    # 存储找到的调用关系
    invokes = []

    # 遍历 AST 节点以找到调用关系
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.FunctionDef) and ast_node.name == caller_function_name:
            # 遍历函数体内的所有子节点
            for sub_node in ast.walk(ast_node):
                # 检查子节点是否为函数调用
                if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                    called_function = sub_node.func.id
                    if called_function in callee_function_names:
                        invokes.append(called_function)
                # 检查子节点是否在循环中
                if isinstance(sub_node, (ast.For, ast.While)):
                    for loop_node in ast.walk(sub_node):
                        if isinstance(loop_node, ast.Call) and isinstance(loop_node.func, ast.Name):
                            called_function = loop_node.func.id
                            if called_function in callee_function_names:
                                invokes.append(called_function)
                # 检查子节点是否在异常处理块中
                if isinstance(sub_node, (ast.Try, ast.ExceptHandler)):
                    for try_node in ast.walk(sub_node):
                        if isinstance(try_node, ast.Call) and isinstance(try_node.func, ast.Name):
                            called_function = try_node.func.id
                            if called_function in callee_function_names:
                                invokes.append(called_function)
                # 检查子节点是否在条件表达式中
                if isinstance(sub_node, ast.IfExp):
                    for ifexp_node in ast.walk(sub_node):
                        if isinstance(ifexp_node, ast.Call) and isinstance(ifexp_node.func, ast.Name):
                            called_function = ifexp_node.func.id
                            if called_function in callee_function_names:
                                invokes.append(called_function)
                # 检查子节点是否作为返回值
                if isinstance(sub_node, ast.Return):
                    for return_node in ast.walk(sub_node):
                        if isinstance(return_node, ast.Call) and isinstance(return_node.func, ast.Name):
                            called_function = return_node.func.id
                            if called_function in callee_function_names:
                                invokes.append(called_function)
                # 检查子节点是否在列表或字典解析中
                if isinstance(sub_node, (ast.ListComp, ast.DictComp)):
                    for comp_node in ast.walk(sub_node):
                        if isinstance(comp_node, ast.Call) and isinstance(comp_node.func, ast.Name):
                            called_function = comp_node.func.id
                            if called_function in callee_function_names:
                                invokes.append(called_function)

    # 在图中添加调用关系的边
    for callee in invokes:
        callee_node = next(func_node for func_node in func_nodes if func_node.endswith(f":{callee}"))
        graph.add_edge(node, callee_node, type='invokes')


def add_edges(graph):
    for node, attributes in graph.nodes(data=True):
        if attributes.get('type') == 'class':
            analyze_for_class_node(node, graph)
        elif attributes.get('type') == 'function':
            analyze_for_function_node(node, graph)


def visualize_graph(graph, show_classes=False, show_functions=False, max_nodes=20):
    if not show_classes or not show_functions:
        nodes_to_remove = [node for node, data in graph.nodes(data=True)
                           if (not show_classes and data['type'] == 'class') or
                           (not show_functions and data['type'] == 'function')]
        graph.remove_nodes_from(nodes_to_remove)

    if len(graph.nodes) > max_nodes:
        subgraph_nodes = list(graph.nodes)[:max_nodes]
        subgraph = graph.subgraph(subgraph_nodes).copy()
    else:
        subgraph = graph

    pos = nx.shell_layout(subgraph)
    edge_labels = {(u, v): d['type'] for u, v, d in subgraph.edges(data=True)}
    node_labels = {n: n for n in subgraph.nodes()}

    # Create color maps for nodes and edges based on type
    node_colors = []
    for n, data in subgraph.nodes(data=True):
        if data['type'] == 'class':
            node_colors.append('lightgreen')
        elif data['type'] == 'function':
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgrey')

    edge_colors = []
    for u, v, data in subgraph.edges(data=True):
        if data['type'] == 'contains':
            edge_colors.append('skyblue')
        elif data['type'] == 'imports':
            edge_colors.append('forestgreen')
        else:
            edge_colors.append('magenta')

    plt.figure(figsize=(12, 12))
    # Draw nodes with specified colors
    nx.draw(subgraph, pos, with_labels=True, labels=node_labels, node_size=3000, node_color=node_colors,
            font_size=10)
    # Draw edges with specified colors
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)
    plt.show()


# 将 NetworkX 图非import、contains的边类型修改为invokes
def convert_edges_to_invokes(G):
    # 找到并修改特定类型的边
    edges_to_modify = []
    for u, v, data in G.edges(data=True):
        if data['type'] in ['composes', 'uses constructor', 'uses helper', 'uses decorator']:
            edges_to_modify.append((u, v))
    # 删除旧边，添加新边
    for u, v in edges_to_modify:
        data = G.get_edge_data(u, v)
        G.remove_edge(u, v)
        data['type'] = 'invokes'
        G.add_edge(u, v, **data)

    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', type=str, default='DATA/repo/astropy__astropy-12907')
    args = parser.parse_args()

    # Generate Dependency Graph
    graph = build_graph(args.repo_path)
    add_edges(graph)
    graph = convert_edges_to_invokes(graph)

    edge_types = []
    for u, v, data in graph.edges(data=True):
        edge_types.append(data['type'])
    print(Counter(edge_types))

    node_types = []
    for node, data in graph.nodes(data=True):
        node_types.append(data['type'])
    print(Counter(node_types))