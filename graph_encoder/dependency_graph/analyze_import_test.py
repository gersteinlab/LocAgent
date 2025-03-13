import os
import sys
import ast
import importlib
import inspect


def find_import_statements(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=file_path)
    import_statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_statements.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            for alias in node.names:
                import_statements.append((module_name, alias.name))
    return import_statements


def main():
    # Path to file B
    file_b_path = '/home/gangda/workspace/auto-search-agent/graph_encoder/DATA/repo/astropy__astropy-12907/astropy/cosmology/connect.py'  # Replace with the actual path to file B
    file_b_dir = os.path.dirname(os.path.abspath(file_b_path))

    # Adjust sys.path
    if file_b_dir not in sys.path:
        sys.path.insert(0, '/home/gangda/workspace/auto-search-agent/graph_encoder/DATA/repo/astropy__astropy-12907')

    # Extract imports from file B
    imports = find_import_statements(file_b_path)

    # Process imports
    for module_name, object_name in imports:
        try:
            if module_name is None or module_name.startswith('.'):
                continue  # Handle or skip relative imports as needed

            # Attempt to import the module
            module = importlib.import_module(module_name)

            # Get the file path
            if object_name:
                # Get the object from the module
                obj = getattr(module, object_name)
                obj_file = inspect.getfile(obj)
                print(f"The object '{module_name}.{object_name}' is located at: {obj_file}")
            else:
                module_file = inspect.getfile(module)
                print(f"The module '{module_name}' is located at: {module_file}")
        except Exception as e:
            print(f"Could not resolve '{module_name}.{object_name or ''}': {e}")


if __name__ == '__main__':
    main()
