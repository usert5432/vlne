"""
A collection of helper functions for `Args` construction.
"""

import copy
import re

def modify_vars(vars_orig, var_modifiers):
    """Return a copy of `vars_orig` modified according to `var_modifiers`.

    Parameters
    ----------
    vars_orig : list of str
        A list of variable names.
    var_modifiers : list of str or None, optional
        A list of variable modifiers. Each variable modifier is a string
        either "+var_name" or "-var_name".
        If `var_modifiers` has "+var_name", then "var_name" will be added
        to the result.
        If `var_modifiers` has "-var_name", then "var_name" will be removed
        from  the result.
        If `var_modifiers` is None, then a simple copy of `vars_orig` will be
        returned.
        Default: None.

    Returns
    -------
    list of str
        A copy of `vars_orig` modified according to `var_modifiers`.
    """

    if var_modifiers is None:
        return vars_orig

    result = vars_orig[:]

    regexp = re.compile(r'([+-])(.*)')

    for vm in var_modifiers:

        res = regexp.match(vm)
        if not res:
            raise ValueError('Failed to parse column modifier %s' % vm)

        mod = res.group(1)
        val = res.group(2)

        if mod == '+':
            if val not in result:
                result.append(val)
        else:
            result.remove(val)

    return result

def update_kwargs(kwargs, extra_kwargs):
    """Modify kwargs dict inplace by items from extra_kwargs.

    This function overrides/updates items from `kwargs` by the items from
    `extra_kwargs`.

    Notes
    -----
    If `extra_kwargs` is None, then `kwargs` will not be modified.

    If `extra_kwargs` contains an item (`key`, `value`) and `kwargs` also
    contains an item (`key`, `orig_value`) and both `value` and `orig_value`
    are dict themselves, then `update_kwargs` will be called recursively
    on `orig_value`. That is `update_kwargs`('orig_value`, `value`) will be
    called.

    Otherwise, for all other (`key`, `value`) pairs in the `extra_kwargs` the
    following will be executed: `kwargs`[`key`] = `value`
    """

    if extra_kwargs is None:
        return

    for k,v in extra_kwargs.items():
        if (
                isinstance(v, dict)
            and k in kwargs
            and isinstance(kwargs[k], dict)
        ):
            update_kwargs(kwargs[k], v)
        else:
            kwargs[k] = copy.deepcopy(v)

def join_dicts(*dicts_list):
    """Return a dict obtained by joining dicts_list with `update_kwargs`."""

    base_dict = {}

    for d in dicts_list:
        update_kwargs(base_dict, d)

    return base_dict

