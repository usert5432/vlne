import copy
from typing import Any, Union, Dict, Tuple

Spec = Union[str, Dict[str, Any]]

def unpack_name_args(obj : Spec) -> Tuple[str, Dict[str, Any]]:
    if isinstance(obj, str):
        return (obj, {})

    obj  = copy.deepcopy(obj)
    name = obj.pop('name')

    if 'kwargs' in obj:
        obj = obj['kwargs']

    return (name, obj)

