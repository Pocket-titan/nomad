from typing import Union, Any
from astropy.time import Time, TimeDelta
import numpy as np


def get_dates(x, number_of_legs):
    J2000 = Time("J2000", format="jyear_str")

    dates = (
        J2000 + TimeDelta(np.cumsum(x, axis=-1)[..., : number_of_legs + 1], format="sec")
    ).to_value("datetime")

    return dates


def get_departure_arrival_time(x, number_of_legs):
    fn = np.vectorize(lambda x: x.strftime("%Y-%m-%d %H:%M"))
    return fn(get_dates(x, number_of_legs)[..., [0, -1]])


def merge(a: dict, b: dict, path=[]):
    for k in b:
        if k not in a:
            a[k] = b[k]
        else:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                merge(a[k], b[k], path + [str(k)])
            elif isinstance(a[k], (list, np.ndarray)) and isinstance(
                b[k], (list, np.ndarray)
            ):
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

        v = dictify(
            dict(enumerate(x)), level=level + 1, maxlevel=maxlevel, leavelast=leavelast
        )

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


def once(fn):
    def inner(*args, **kwargs):
        if not hasattr(inner, "done"):
            inner.done = True
            return fn(*args, **kwargs)

    return inner


def is_notebook():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def truncate(x: Any):
    if isinstance(x, dict):
        return {k: truncate(v) for k, v in x.items()}
    if isinstance(x, list):
        return [truncate(v) for v in x]
    if isinstance(x, tuple):
        return tuple([truncate(v) for v in x])
    if isinstance(x, float):
        ret = str(x).split(".")

        if len(ret) == 1:
            return x

        leading, decimals = ret

        if leading.startswith("0"):
            return round(x, 6)

        return round(x, 2)
    return x
