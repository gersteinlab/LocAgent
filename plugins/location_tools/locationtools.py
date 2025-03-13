from inspect import signature

from plugins.location_tools import repo_ops, retriever
from plugins.location_tools.utils.dependency import import_functions

# import_functions(
#     module=retriever, function_names=retriever.__all__, target_globals=globals()
# )

import_functions(
    module=repo_ops, function_names=repo_ops.__all__, target_globals=globals()
)
__all__ = repo_ops.__all__ # + retriever.__all__

DOCUMENTATION = ''
for func_name in __all__:
    func = globals()[func_name]

    cur_doc = func.__doc__
    # remove indentation from docstring and extra empty lines
    cur_doc = '\n'.join(filter(None, map(lambda x: x.strip(), cur_doc.split('\n'))))
    # now add a consistent 4 indentation
    cur_doc = '\n'.join(map(lambda x: ' ' * 4 + x, cur_doc.split('\n')))

    fn_signature = f'{func.__name__}' + str(signature(func))
    DOCUMENTATION += f'{fn_signature}:\n{cur_doc}\n\n'
