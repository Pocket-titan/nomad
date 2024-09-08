from typing import Optional, Callable
from itertools import groupby
from wurlitzer import pipes
from functools import wraps
from pathlib import Path

import numpy as np
import traceback
import logging
import re
import os

from tudatpy.astro.time_conversion import DateTime
from tudatpy.trajectory_design.transfer_trajectory import (
    TransferTrajectory,
    TransferLegSettings,
    TransferNodeSettings,
    print_parameter_definitions,
)

from trajectory.lib.utils import flatten, dictify

ROOT_FOLDER = str(Path(__file__).parents[2].absolute())
logger = logging.getLogger(os.environ.get("SUBP_LOG_NAME", None))


bound_map = {
    "time": [DateTime(2020, 1, 1).epoch(), DateTime(2050, 1, 1).epoch()],
    "tof_fraction": [0, 1],
    "velocity_magnitude": [0, 1e6],
    "in-plane_angle": [0, 2 * np.pi],
    "out-of-plane_angle": [-np.pi / 2, np.pi / 2],
    "periapsis": [1e5, 1e26],
    "delta_v": [0, 10_000],
    "number_of_revolutions": [0, 5],
}

dtype_map = {
    "time": "float",
    "tof_fraction": "float",
    "velocity_magnitude": "float",
    "in-plane_angle": "float",
    "out-of-plane_angle": "float",
    "periapsis": "float",
    "delta_v": "float",
    "number_of_revolutions": "int",
}

parameter_map = {
    "Time": {
        "time": "time",
    },
    "Node": {
        "Swingby periapsis": "periapsis",
        "Swingby Delta V": "delta_v",
        "Outgoing excess velocity magnitude": "velocity_magnitude",
        "Outgoing excess velocity in-plane angle": "in-plane_angle",
        "Outgoing excess velocity out-of-plane angle": "out-of-plane_angle",
        "Swingby orbital plane angle (with respect to the incoming velocity and node velocity)": "in-plane_angle",
        "Incoming excess velocity magnitude": "velocity_magnitude",
        "Incoming excess velocity in-plane angle": "in-plane_angle",
        "Incoming excess velocity out-of-plane angle": "out-of-plane_angle",
    },
    "Leg": {
        "DSM (velocity-based) Time-of-flight fraction": "tof_fraction",
        "Number of revolutions (integer number >= 0)": "number_of_revolutions",
    },
}


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


@once
def log_once(*args):
    logger.info(*args)


class Problem:
    def __init__(
        self,
        create_obj: Callable[
            ...,
            tuple[
                TransferTrajectory,
                TransferLegSettings,
                TransferNodeSettings,
                Optional[np.ndarray],
            ],
        ],
        bounds={},
        fixed={},
        dim=1,
        cache_eq=lambda old, new: True,
    ) -> None:
        def create_wrapper(fn):
            @wraps(fn)
            def wrapper(*args):
                og_dv = args
                if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
                    og_dv = args[0]

                crargs = og_dv if len(og_dv) == 0 else self._add_info(og_dv)
                ret = fn(*crargs)

                if len(ret) == 4:
                    obj, leg_settings, node_settings, dv = ret
                    return obj, leg_settings, node_settings, dv

                obj, leg_settings, node_settings = ret
                return obj, leg_settings, node_settings, og_dv

            return wrapper

        self.haslogged = False
        self.create_obj = create_wrapper(create_obj)
        self.cache_eq = cache_eq
        self.errs = 0
        self.dim = dim

        self.bounds = bounds
        self.fixed = fixed

        self._set_parameters()

    def get_nobj(self) -> int:
        return self.dim

    def get_nix(self) -> int:
        return len(
            [x for x in self.parameters if "value" not in x and x["dtype"] == "int"]
        )

    def get_bounds(self):
        def map_bounds(x):
            if "bounds" in x:
                return x

            return {**x, "bounds": bound_map[x["name"]]}

        bounds = self._preprocess(
            [map_bounds(x) for x in self.parameters if "value" not in x]
        )
        return tuple([*(list(x) for x in zip(*[x["bounds"] for x in bounds]))])

    def fitness(self, dv: np.ndarray):
        obj, dv = self._obj_dv(*dv)

        try:
            obj.evaluate(*self.convert_parameters(dv))
            delta_v = obj.delta_v
        except Exception:
            if self.errs == 0:
                tb = " ".join(
                    x.strip()
                    .replace(ROOT_FOLDER, ".")
                    .replace(" (most recent call last)", "")
                    for x in traceback.format_exc().split("\n")
                )
                logger.warning(tb)
            self.errs += 1
            return self._death_penalty()

        if self.dim == 1:
            return [delta_v]

        tof = obj.time_of_flight
        return [delta_v, tof]

    def evaluate(self, dv: np.ndarray):
        return self.obj(*dv).evaluate(*self.convert_parameters(dv))

    def convert_parameters(self, dv: np.ndarray):
        pars = self._add_info(dv)

        def pick(x):
            return x["value"]

        departure = [pick(x) for x in pars if x["kind"] == "departure"]
        leg_tof = [pick(x) for x in pars if x["kind"] == "leg_tof"]
        node_times = np.cumsum([*departure, *leg_tof])

        node_parameters = [
            [pick(x) for x in pars if x["kind"] == "node_parameters" and x["nr"] == i]
            for i in range(self.number_of_nodes)
        ]

        leg_parameters = [
            [pick(x) for x in pars if x["kind"] == "leg_parameters" and x["nr"] == i]
            for i in range(self.number_of_legs)
        ]

        return node_times, leg_parameters, node_parameters

    def obj(self, *dv):
        return self._obj_dv(*dv)[0]

    @property
    def death_value(self) -> float:
        return 1e16

    def _death_penalty(self) -> list[float]:
        return [self.death_value] * self.dim

    def _obj_dv(self, *dv):
        if hasattr(self, "_obj") and hasattr(self, "_last_pars"):
            if self.cache_eq(self._last_pars, self._add_info(dv) if len(dv) > 0 else dv):
                return self._obj(), dv

        obj, _, _, dv = self.create_obj(*dv)
        self._obj = lambda: obj
        self._last_pars = self._add_info(dv)

        return self._obj(), dv

    def _add_info(self, dv: np.ndarray):
        indices = [i for i, x in enumerate(self.parameters) if "value" not in x]
        key = {i: j for i, j in zip(indices, np.arange(len(dv)))}

        def map_value(i, x):
            if i in indices:
                return {**x, "value": dv[key[i]]}

            return x

        return self._postprocess(
            [map_value(i, x) for i, x in enumerate(self._preprocess(self.parameters))]
        )

    def _set_parameters(self):
        obj, leg_settings, node_settings, _ = self.create_obj()
        self.number_of_nodes = obj.number_of_nodes
        self.number_of_legs = obj.number_of_legs

        out = get_parameter_definitions(leg_settings, node_settings)

        def parse(x):
            match1 = re.match(r"Parameter ([0-9]*): (Node) (time) ([0-9]*)", x)

            if match1:
                idx, sort, name, nr = match1.groups()
                return int(idx), "Time", int(nr), name

            match2 = re.match(r"Parameter ([0-9]*): (Node|Leg)\s+([0-9]*)?\s+(.*)", x)

            if match2:
                idx, sort, nr, name = match2.groups()
                return int(idx), sort, int(nr), name

            raise Exception()

        def map_parameter(x):
            idx, sort, nr, _name = x
            name = parameter_map[sort][_name]

            if sort == "Node":
                kind = "node_parameters"
            if sort == "Leg":
                kind = "leg_parameters"
            if sort == "Time":
                if nr == 0:
                    kind = "departure"
                else:
                    kind = "leg_tof"
                    nr -= 1

            return {
                "i": idx,
                "kind": kind,
                "nr": nr,
                "name": name,
                "dtype": dtype_map[name],
            }

        parameters = [map_parameter(parse(x)) for x in out.split("\n")]

        js = [(x["kind"], x["nr"]) for x in parameters]
        js = flatten([[*[i for i, _ in enumerate(v)]] for k, v in groupby(js)])

        for i, j in enumerate(js):
            parameters[i]["j"] = j

        def update(partial: dict, update: dict):
            idx = [
                i
                for i, x in enumerate(parameters)
                if all([k in x and x[k] == v for k, v in partial.items()])
            ]

            if len(idx) > 0:
                parameters[idx[0]].update(update)

        for k, seq in enumerate([self.fixed, self.bounds]):
            key = ["value", "bounds"][k]

            for kind, v in seq.items():
                if isinstance(v, (list, np.ndarray)) and np.shape(v) == (2,):
                    v = np.reshape(v, (1, 2))

                for nr, val in dictify(v, leavelast=k == 1).items():
                    if isinstance(val, dict):
                        for j, val2 in val.items():
                            update({"kind": kind, "nr": nr, "j": j}, {key: val2})
                    else:
                        update({"kind": kind, "nr": nr}, {key: val})

        self.parameters = parameters

    def _preprocess(self, pars):
        def transform(x):
            if x["kind"] == "node_parameters" and x["name"] == "periapsis":
                f = np.log10

                return {
                    **x,
                    **({"value": f(x["value"])} if "value" in x else {}),
                    **({"bounds": f(x["bounds"])} if "bounds" in x else {}),
                }

            return x

        return [transform(x) for x in pars]

    def _postprocess(self, pars):
        def transform(x):
            if x["kind"] == "node_parameters" and x["name"] == "periapsis":
                f = lambda y: np.power(10, y)  # noqa: E731

                return {
                    **x,
                    **({"value": f(x["value"])} if "value" in x else {}),
                    **({"bounds": f(x["bounds"])} if "bounds" in x else {}),
                }

            if x["dtype"] == "int":
                return {
                    **x,
                    **({"value": int(x["value"])} if "value" in x else {}),
                    **(
                        {"bounds": np.array(x["bounds"], dtype=int)}
                        if "bounds" in x
                        else {}
                    ),
                }

            return x

        return [transform(x) for x in pars]

    def __getstate__(self):
        out = self.__dict__.copy()
        if "_obj" in out:
            del out["_obj"]
        if "_last_pars" in out:
            del out["_last_pars"]
        return out
