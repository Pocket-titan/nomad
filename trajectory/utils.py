from typing import Union, Any
import numpy as np
from wurlitzer import pipes
from tudatpy.trajectory_design.transfer_trajectory import print_parameter_definitions


def merge(a: dict, b: dict, path=[]):
    for k in b:
        if k not in a:
            a[k] = b[k]
        else:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                merge(a[k], b[k], path + [str(k)])
            elif isinstance(a[k], (list, np.ndarray)) and isinstance(b[k], (list, np.ndarray)):
                a[k] = b[k]
            elif a[k] != b[k]:
                raise Exception("Conflict at " + ".".join(path + [str(k)]))

    return a


def dictify(x: Union[dict, list, np.ndarray, Any], level=-1, maxlevel=2, leavelast=True):
    if level > maxlevel:
        return x

    if isinstance(x, dict):
        return {
            k: dictify(v, level=level + 1, maxlevel=maxlevel, leavelast=leavelast)
            for k, v in x.items()
        }
    elif isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return {}

        v = dictify(dict(enumerate(x)), level=level + 1, maxlevel=maxlevel, leavelast=leavelast)

        if (
            leavelast
            and isinstance(v, dict)
            and not isinstance([*v.values()][0], (list, dict, np.ndarray))
        ):
            return x

        return v

    return x


def undictify(x):
    if isinstance(x, dict):
        if all([isinstance(y, (list, np.ndarray)) or y is None for y in x.values()]):
            return [*x.values()]

        return undictify({k: undictify(v) for k, v in x.items()})

    return x


def flatten(x):
    flat = []

    for y in x:
        if isinstance(y, list) and len(y) > 0:
            flat.extend(flatten(y))
        else:
            flat.append(y)

    return flat


def get_parameter_definitions(leg_settings, node_settings):
    with pipes() as (out, err):
        print_parameter_definitions(leg_settings, node_settings)

    stdout = out.read()
    return stdout.strip()


def once(fn):
    def inner(*args, **kwargs):
        if not hasattr(inner, "done"):
            inner.done = True
            return fn(*args, **kwargs)

    return inner
