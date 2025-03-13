import libcst as cst
import libcst.matchers as m


class ClassDocstringVisitor(cst.CSTVisitor):
    def __init__(self):
        self.class_docstrings = []

    def visit_ClassDef(self, node: cst.ClassDef):
        # Extract the docstring if it's the first statement in the body
        if node.body.body and isinstance(node.body.body[0], cst.SimpleStatementLine):
            first_stmt = node.body.body[0].body[0]
            if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
                # This is the docstring
                self.class_docstrings.append(first_stmt.value.value)


def parse_class_docstrings(file_content: str) -> list:
    """Parse the docstrings of classes in the given code."""
    try:
        tree = cst.parse_module(file_content)
    except:
        return []

    visitor = ClassDocstringVisitor()
    tree.visit(visitor)
    return visitor.class_docstrings