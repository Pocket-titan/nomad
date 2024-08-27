# %%
from typing import Union, Any, Callable, TypeVar
from astropy.time import Time, TimeDelta
from functools import reduce
from pprint import pprint
from copy import copy, deepcopy

import pygmo_plugins_nonfree as ppnf
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pygmo as pg
import re

from tudatpy.trajectory_design.transfer_trajectory import TransferTrajectory
from tudatpy.trajectory_design import transfer_trajectory, shape_based_thrust
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.math.root_finders import secant
from tudatpy import constants

from trajectory.utils import dictify, get_parameter_definitions

np.set_printoptions(precision=2, suppress=True)
sns.set_theme(style="ticks", palette="Set2")
pal = sns.color_palette("Set2")


# %%

bound_map = {
    "tof_fraction": [0, 1],
    "velocity_magnitude": [0, 1e6],
    "in-plane_angle": [0, 2 * np.pi],
    "out-of-plane_angle": [-np.pi / 2, np.pi / 2],
    "periapsis": [1e5, 1e26],
    "delta_v": [0, 10_000],
    "number_of_revolutions": [0, 5],
}

type_map = {
    "tof_fraction": "float",
    "velocity_magnitude": "float",
    "in-plane_angle": "float",
    "out-of-plane_angle": "float",
    "periapsis": "float",
    "delta_v": "float",
    "number_of_revolutions": "int",
}

class_map = {
    "float": float,
    "int": int,
}

parameter_map = {
    "leg": {
        "velocity-based DSM": {
            1: ["tof_fraction"],
        },
        "hodographic leg": {
            1: ["number_of_revolutions"],
        },
    },
    "node": {
        "escape_and_departure": {
            3: [
                "velocity_magnitude",
                "in-plane_angle",
                "out-of-plane_angle",
            ],
        },
        "swingby": {
            3: [
                "periapsis",
                "in-plane_angle",
                "delta_v",
            ],
            6: [
                "velocity_magnitude",
                "in-plane_angle",
                "out-of-plane_angle",
                "periapsis",
                "in-plane_angle",
                "delta_v",
            ],
        },
        "capture_and_insertion": {
            3: [
                "velocity_magnitude",
                "in-plane_angle",
                "out-of-plane_angle",
            ]
        },
    },
}


def match_error_message(err):
    match1 = re.match(
        r"Error when getting (?:leg|node) parameters for (leg|node) ([0-9]*) \(type: (.*)\).*\. (?:Leg|Node) should have ([0-9]*) free parameters",
        str(err),
    )

    if match1:
        sort, idx, _type, npar = match1.groups()
        return sort, idx, _type, npar

    match2 = re.match(
        r"Error getting (?:leg|node) parameters, (.*) free parameters are incompatible for (leg|node) ([0-9]*)\. Expected ([0-9]*) free parameters",
        str(err),
    )

    if match2:
        _type, sort, idx, npar = match2.groups()
        return sort, idx, _type, npar

    raise Exception("Error parsing error message:", err)


T = TypeVar("T")


class Problem:
    def __init__(
        self,
        create_obj: Callable[[T], TransferTrajectory],
        bounds={},
        fixed={},
        dim=1,
    ) -> None:
        def create_wrapper(*args):
            og_dv = args
            if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
                og_dv = args[0]

            ret = create_obj(*og_dv)

            if isinstance(ret, (tuple, list)) and len(ret) == 2:
                obj, dv = ret
                return obj, dv

            obj = ret
            return obj, og_dv

        self.create_obj = create_wrapper
        self.dim = dim

        self.bounds = bounds
        self.fixed = fixed

        self._determine_parameters()
        self._set_parameters()

    def get_nobj(self) -> int:
        return self.dim

    def get_nix(self) -> int:
        return len([*filter(lambda x: x["type"] == "int" and "value" not in x, self.parameters)])

    def get_bounds(self):
        defaults = {
            "departure": [DateTime(2024, 1, 1).epoch(), DateTime(2040, 1, 1).epoch()],
            "leg_tof": [1 * constants.JULIAN_DAY, 10 * constants.JULIAN_YEAR],
        }

        def map_bound(x: dict):
            if "value" in x:
                return None

            if "bounds" in x:
                return x["bounds"]

            val = None
            if x["kind"] in defaults:
                val = defaults[x["kind"]]
            elif "name" in x:
                val = bound_map[x["name"]]

            if val is None:
                raise Exception(f"No bounds found for parameter: {x}")
            return np.array(val, dtype=class_map[x["type"]])

        pars = copy(self.parameters)
        for p in pars:
            p["bounds"] = map_bound(p)
        pars = self._preprocess([*filter(lambda x: x["bounds"] is not None, pars)])

        bounds = map(lambda x: x["bounds"], pars)
        return tuple([*(list(x) for x in zip(*bounds))])

    def fitness(self, dv: np.ndarray):
        obj, dv = self._obj_dv(dv)

        try:
            obj.evaluate(*self._convert_parameters(dv))
            delta_v = obj.delta_v
        except Exception as e:
            # print("Err:", e)
            return self._death_penalty()

        if self.dim == 1:
            return [delta_v]

        tof = obj.time_of_flight
        return [delta_v, tof]

    def evaluate(self, *args):
        if len(args) == 1:
            obj, dv = self._obj_dv(*args)
            return obj.evaluate(*self._convert_parameters(dv))

        return self.obj().evaluate(*args)

    def _obj_dv(self, *dv):
        if len(dv) > 0 or not hasattr(self, "_obj"):
            obj, dv = self.create_obj(*dv)
            self._obj = lambda: obj
            return self._obj(), dv

        return self._obj(), []

    def obj(self, *dv):
        return self._obj_dv(*dv)[0]

    @property
    def death_value(self) -> float:
        return 1e16

    def _death_penalty(self) -> list[float]:
        return [self.death_value] * self.dim

    def _convert_parameters(self, dv: np.ndarray):
        pars = [*filter(lambda x: x["kind"] != "obj", self._preprocess(deepcopy(self.parameters)))]
        indices = [i for i, x in enumerate(pars) if "value" not in x]
        for i, x in zip(indices, dv):
            pars[i]["value"] = x
        assert len([*filter(lambda x: "value" not in x, pars)]) == 0

        pars = self._postprocess(pars)

        def pick(x):
            return x["value"]

        departure = [*map(pick, filter(lambda x: x["kind"] == "departure", pars))]
        leg_tof = [*map(pick, filter(lambda x: x["kind"] == "leg_tof", pars))]
        node_times = np.cumsum([*departure, *leg_tof])

        node_parameters = [
            [*map(pick, filter(lambda x: x["kind"] == "node_parameters" and x["nr"] == i, pars))]
            for i in range(len(self.nodes))
        ]

        leg_parameters = [
            [*map(pick, filter(lambda x: x["kind"] == "leg_parameters" and x["nr"] == i, pars))]
            for i in range(len(self.legs))
        ]

        return node_times, leg_parameters, node_parameters

    def _determine_parameters(self):
        args = []
        if "obj" in self.bounds:
            args = [x[0] for x in self.bounds["obj"]]
        obj, _ = self.create_obj(*args)

        nodes = [("node", None, 0)] * obj.number_of_nodes
        legs = [("leg", None, 0)] * obj.number_of_legs

        node_times = np.linspace(1e3, 1e4, num=obj.number_of_nodes)
        leg_parameters = [[] for _ in range(obj.number_of_legs)]
        node_parameters = [[] for _ in range(obj.number_of_nodes)]

        def try_evaluate(*parameters):
            try:
                obj.evaluate(*parameters)
            except RuntimeError as e:
                sort, idx, _type, npar = match_error_message(e)

                if sort == "leg":
                    arr = parameters[1]
                    kind = legs
                if sort == "node":
                    arr = parameters[2]
                    kind = nodes

                pars = parameter_map[sort][_type][int(npar)]
                vals = [bound_map[x][0] + (bound_map[x][1] - bound_map[x][0]) / 4 for x in pars]
                vals = [(class_map[type_map[x]])(y) for x, y in zip(pars, vals)]

                kind[int(idx)] = (sort, _type, int(npar))
                arr[int(idx)].extend(vals)

                return try_evaluate(*parameters)

        try_evaluate(node_times, leg_parameters, node_parameters)
        self.nodes = nodes
        self.legs = legs

    def _set_parameters(self):
        floats = []
        ints = []

        args = []
        if "obj" in self.bounds:
            args = [x[0] for x in self.bounds["obj"]]
        obj, _ = self.create_obj(*args)

        if "obj" in self.bounds:
            for nr, x in enumerate(self.bounds["obj"]):
                if isinstance(x, np.ndarray):
                    if x.dtype.kind == "f":
                        floats.append(dict(nr=nr, i=0, kind="obj"))
                    elif x.dtype.kind == "i":
                        ints.append(dict(nr=nr, i=0, kind="obj"))
                else:
                    floats.append(dict(nr=nr, i=0, kind="obj"))

        floats.append(dict(nr=0, i=0, kind="departure"))
        floats.extend([dict(nr=nr, i=0, kind="leg_tof") for nr in range(obj.number_of_legs)])

        for j, seq in enumerate([self.nodes, self.legs]):
            kind = ["node_parameters", "leg_parameters"][j]

            for nr, (sort, _type, npar) in enumerate(seq):
                if _type is None:
                    continue

                for i, par in enumerate(parameter_map[sort][_type][npar]):
                    arr = ints if type_map[par] == "int" else floats
                    arr.append(dict(nr=nr, i=i, kind=kind, name=par))

        self.parameters = [
            *[{"type": "float", **x} for x in floats],
            *[{"type": "int", **x} for x in ints],
        ]

        def update(partial: dict, update: dict):
            idx = [
                i
                for i, x in enumerate(self.parameters)
                if all([k in x and x[k] == v for k, v in partial.items()])
            ]

            if len(idx) > 0:
                self.parameters[idx[0]].update(update)

        for j, seq in enumerate([self.fixed, self.bounds]):
            key = ["value", "bounds"][j]

            for k, v in seq.items():
                if isinstance(v, (list, np.ndarray)) and np.shape(v) == (2,):
                    v = np.reshape(v, (1, 2))

                for nr, val in dictify(v, leavelast=j == 1).items():
                    if isinstance(val, dict):
                        for i, val2 in val.items():
                            update({"kind": k, "nr": nr, "i": i}, {key: val2})
                    else:
                        update({"kind": k, "nr": nr}, {key: val})

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

            if x["type"] == "int":
                dtype = class_map[x["type"]]
                return {
                    **x,
                    **({"value": dtype(x["value"])} if "value" in x else {}),
                    **({"bounds": np.array(x["bounds"], dtype=dtype)} if "bounds" in x else {}),
                }

            return x

        return [transform(x) for x in pars]

    def __getstate__(self):
        out = self.__dict__.copy()
        if "_obj" in out:
            del out["_obj"]
        return out


central_body = "Sun"

body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = 3.8913e9
arrival_eccentricity = 0.999486

bodies = environment_setup.create_simplified_system_of_bodies()

leg_settings, node_settings = transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
    body_order,
    departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
)

leg_settings, node_settings = transfer_trajectory.mga_settings_dsm_velocity_based_legs(
    body_order,
    departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
)


out = get_parameter_definitions(leg_settings, node_settings)
print(out)


def map_parameter(x):
    match1 = re.match(
        r"Parameter ([0-9]*): (Node) (time) ([0-9]*)",
        str(x),
    )

    if match1:
        idx, sort, name, nr = match1.groups()
        return int(idx), sort, int(nr), name

    match2 = re.match(
        r"Parameter ([0-9]*): (Node|Leg)\s+([0-9]*)?\s+(.*)",
        str(x),
    )

    if match2:
        idx, sort, nr, name = match2.groups()
        return int(idx), sort, int(nr), name

    raise Exception()


[map_parameter(x) for x in out.split("\n")]


# %%
# leg_settings, node_settings = transfer_trajectory.mga_settings_dsm_velocity_based_legs(
#     body_order,
#     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
#     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
# )


# leg_settings, node_settings = (
#     transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(
#         body_order,
#         [25 * constants.JULIAN_DAY, 60 * constants.JULIAN_DAY, 12 * constants.JULIAN_YEAR],
#         [10, 20, 30],
#         departure_orbit=(departure_semi_major_axis, departure_eccentricity),
#         arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
#     )
# )

# leg_settings, node_settings = transfer_trajectory.mga_settings_spherical_shaping_legs(
#     body_order,
#     root_finder_settings=secant(),
#     lower_bound_free_coefficient=0,
#     upper_bound_free_coefficient=1000,
#     initial_value_free_coefficient=-5,
#     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
#     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
# )


def create_transfer_object(*dv) -> TransferTrajectory:
    central_body = "Sun"

    body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = 3.8913e9
    arrival_eccentricity = 0.999486

    bodies = environment_setup.create_simplified_system_of_bodies()

    leg_settings, node_settings = transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )

    leg_settings, node_settings = (
        transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(
            body_order,
            dv[:3],
            dv[3:],
            departure_orbit=(departure_semi_major_axis, departure_eccentricity),
            arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
        )
    )

    return transfer_trajectory.create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    ), dv


node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]

bounds = dict(
    # obj=[],
    obj=[
        [10 * constants.JULIAN_DAY, 18 * constants.JULIAN_YEAR],
        [10 * constants.JULIAN_DAY, 18 * constants.JULIAN_YEAR],
        [10 * constants.JULIAN_DAY, 18 * constants.JULIAN_YEAR],
        np.array([0, 5], dtype=int),
        np.array([0, 5], dtype=int),
        np.array([0, 5], dtype=int),
    ],
    departure=[DateTime(2028, 4, 6).epoch(), DateTime(2033, 12, 31).epoch()],
    leg_tof=np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]],
    node_parameters={},
    leg_parameters={},
)

fixed = dict(
    obj=[],
    departure={},
    leg_tof=[],
    node_parameters={},
    leg_parameters={},
)

p = Problem(
    create_transfer_object,
    bounds=bounds,
    fixed=fixed,
    dim=1,
)

print(pg.problem(p))
# %%


def evolve(
    p: Problem,
    num_generations=100,
    pop_size=10,
    seed=4444,
    **kwargs,
):
    # fmt: off
    if p.dim == 1:
        def gradient(self, dv):
            return pg.estimate_gradient(lambda x: self.fitness(x), dv, 1e-8)

        p.gradient = gradient.__get__(p)
    # fmt: on

    prob = pg.problem(p)

    if p.dim == 1:
        # algo = pg.nlopt(solver="slsqp")
        algo = ppnf.snopt7()
        algo.set_random_sr_seed(seed)
        algo = pg.mbh(algo, stop=5, perturb=0.01, seed=seed)
    if p.dim == 2:
        algo = pg.moead(gen=num_generations, seed=seed, **kwargs)
    algo = pg.algorithm(algo)

    island = pg.island(algo=algo, prob=prob, size=pop_size, seed=seed)

    results = dict(f=[], x=[])

    for i in range(1, num_generations + 1):
        island.evolve()
        island.wait_check()

        results["f"].append(island.get_population().get_f())
        results["x"].append(island.get_population().get_x())

        if i % 5 == 0:
            print(f"Generation {i}/{num_generations}...")

    generations = [[i] * pop_size for i in range(1, num_generations + 1)]
    fs = np.concatenate(results["f"])
    xs = np.concatenate(results["x"])

    tof = np.cumsum(xs[:, : p.obj(xs[0]).number_of_legs + 1], axis=-1)
    tof = (tof[:, -1] - tof[:, 0]) / constants.JULIAN_YEAR

    df = pl.DataFrame(
        {
            "dv": fs[:, 0],
            "tof": tof,
            "x": xs,
            "gen": np.concatenate(generations),
        }
    )
    oglen = len(df)
    df = df.filter((pl.col("dv") != p.death_value))
    newlen = len(df)

    print(f"discarded: {oglen - newlen} out of {oglen}, or: {(oglen - newlen)/oglen * 100:.0f}%")

    return df


df = evolve(
    p,
    num_generations=100,
    pop_size=40,
    seed=4444,
)

# %%
champions = (
    (
        df.filter(pl.col("gen") > 3)
        .group_by("gen")
        .agg(pl.all().sort_by("dv").head(5))
        .explode(pl.all().exclude("gen"))
    )
    .sort("dv")
    .filter(pl.col("dv") < 20_000)
)

plt.scatter(champions["tof"], champions["dv"], c=champions["gen"], cmap="viridis_r")
plt.xlabel("Time of flight [yr]")
plt.ylabel(r"$\Delta$V [m/s]")

# %%
champions = (df.group_by("gen").agg(pl.all().sort_by("dv").first())).sort("gen")
plt.plot(champions["gen"], champions["dv"])
plt.xlabel("Generation")
plt.ylabel(r"$\Delta$V [m/s]")
plt.title("Best individual per generation")

# %%
selection = df.filter(pl.col("dv") <= 15_000)
xs = np.array(selection.select(pl.col("x")).to_series().to_list())

J2000 = Time("J2000", format="jyear_str")

departures = J2000 + TimeDelta(xs[:, 0], format="sec")
arrivals = J2000 + TimeDelta(np.sum(xs[:, : p.obj().number_of_legs + 1], axis=-1), format="sec")

dvs = selection["dv"]

plt.scatter(
    departures.to_value("datetime64"),
    arrivals.to_value("datetime64"),
    cmap="viridis_r",
    c=dvs,
)
plt.colorbar(label=r"$\Delta$V [m/s]")
plt.xticks(rotation=-45)
plt.xlabel("Departure date")
plt.ylabel("Arrival date")


# %%
