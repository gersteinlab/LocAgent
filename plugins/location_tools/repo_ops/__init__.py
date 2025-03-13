from plugins.location_tools.repo_ops import repo_ops
from plugins.location_tools.utils.dependency import import_functions

import_functions(
    module=repo_ops, function_names=repo_ops.__all__, target_globals=globals()
)
__all__ = repo_ops.__all__
