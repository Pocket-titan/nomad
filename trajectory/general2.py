# %%
from typing import Optional, Union, Any, Callable
from astropy.time import Time, TimeDelta
from pprint import pprint
from itertools import groupby

import multiprocessing as mp
import pygmo_plugins_nonfree as ppnf
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pygmo as pg
import time
import re

from tudatpy.trajectory_design.transfer_trajectory import (
    TransferTrajectory,
    TransferLegSettings,
    TransferNodeSettings,
    print_parameter_definitions,
)
from tudatpy.trajectory_design import transfer_trajectory, shape_based_thrust
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.math.root_finders import secant
from tudatpy import constants

from trajectory.utils import flatten, dictify, get_parameter_definitions, once

np.set_printoptions(precision=2, suppress=True)
sns.set_theme(style="ticks", palette="Set2")
pal = sns.color_palette("Set2")

# %%
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
        def create_wrapper(*args):
            og_dv = args
            if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
                og_dv = args[0]

            crargs = og_dv if len(og_dv) == 0 else self._add_info(og_dv)
            ret = create_obj(*crargs)

            if len(ret) == 4:
                obj, leg_settings, node_settings, dv = ret
                return obj, leg_settings, node_settings, dv

            obj, leg_settings, node_settings = ret
            return obj, leg_settings, node_settings, og_dv

        self.create_obj = create_wrapper
        self.cache_eq = cache_eq
        self.errs = 0
        self.dim = dim

        self.bounds = bounds
        self.fixed = fixed

        self._set_parameters()

    def get_nobj(self) -> int:
        return self.dim

    def get_nix(self) -> int:
        return len([x for x in self.parameters if "value" not in x and x["dtype"] == "int"])

    def get_bounds(self):
        def map_bounds(x):
            if "bounds" in x:
                return x

            return {**x, "bounds": bound_map[x["name"]]}

        bounds = self._preprocess([map_bounds(x) for x in self.parameters if "value" not in x])
        return tuple([*(list(x) for x in zip(*[x["bounds"] for x in bounds]))])

    def fitness(self, dv: np.ndarray):
        obj, dv = self._obj_dv(*dv)

        try:
            obj.evaluate(*self.convert_parameters(dv))
            delta_v = obj.delta_v
        except Exception as e:
            # print("Err:", e)
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
        from trajectory.utils import get_parameter_definitions

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
                    **({"bounds": np.array(x["bounds"], dtype=int)} if "bounds" in x else {}),
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


@once
def print_parameters(leg_settings, node_settings):
    print_parameter_definitions(leg_settings, node_settings)


def create_obj(*pars):
    # if len(pars) == 0:
    #     leg_tof = [1, 1, 1]
    #     num_revolutions = [1, 1, 1]
    # else:
    #     leg_tof = [x["value"] for x in pars if x["kind"] == "leg_tof"]
    #     num_revolutions = [x["value"] for x in pars if x["name"] == "number_of_revolutions"]

    central_body = "Sun"

    body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = 3.8913e9
    arrival_eccentricity = 0.999486

    bodies = environment_setup.create_simplified_system_of_bodies()

    # leg_settings, node_settings = transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
    #     body_order,
    #     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    # )
    leg_settings, node_settings = transfer_trajectory.mga_settings_dsm_velocity_based_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
    # leg_settings, node_settings = (
    #     transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(
    #         body_order,
    #         leg_tof,
    #         num_revolutions,
    #         departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #         arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    #     )
    # )

    print_parameters(leg_settings, node_settings)

    obj = transfer_trajectory.create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )

    return obj, leg_settings, node_settings


node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]


bounds = dict(
    departure=[DateTime(2028, 4, 6).epoch(), DateTime(2033, 12, 31).epoch()],
    leg_tof=np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]],
    node_parameters={},
    leg_parameters={},
)

fixed = dict(
    departure=[],
    leg_tof=[],
    node_parameters={
        1: {2: 0},
        0: {2: 0},
    },
    leg_parameters={},
)


def cache_eq(old, new):
    def getcmp(pars):
        return [
            x["value"]
            for x in old
            if x["kind"] == "leg_tof" or x["name"] == "number_of_revolutions"
        ]

    return np.allclose(getcmp(old), getcmp(new))


p = Problem(
    create_obj,
    # cache_eq=cache_eq,
    bounds=bounds,
    fixed=fixed,
    dim=1,
)


def evolve(
    p: Problem,
    num_evolutions=50,
    num_generations=10,
    pop_size=10,
    seed=4444,
    algo=None,
    **kwargs,
):
    # fmt: off
    if p.dim == 1:
        def gradient(self, dv):
            return pg.estimate_gradient(lambda x: self.fitness(x), dv, 1e-8)

        p.gradient = gradient.__get__(p)
    # fmt: on

    prob = pg.problem(p)

    if algo is None:
        if p.dim == 1:
            # algo = pg.nlopt(solver="slsqp")
            algo = ppnf.snopt7()
            # algo = pg.pso(gen=num_generations, seed=seed)

            if hasattr(algo, "set_random_sr_seed"):
                algo.set_random_sr_seed(seed)

            algo = pg.mbh(algo, stop=5, perturb=0.01, seed=seed)

        if p.dim == 2:
            algo = pg.moead(gen=num_generations, seed=seed, **kwargs)

    algo = pg.algorithm(algo)

    num_islands = mp.cpu_count()
    archi = pg.archipelago(n=num_islands, algo=algo, prob=prob, pop_size=pop_size, seed=seed)

    results = dict(f=[], x=[])
    errs = []

    width = len(str(num_evolutions))
    t0 = time.perf_counter()

    for i in range(1, num_evolutions + 1):
        archi.evolve()
        archi.wait()

        best = np.inf

        for island in archi:
            pop = island.get_population()

            results["f"].append(pop.get_f())
            results["x"].append(pop.get_x())
            errs.append(pop.problem.extract(Problem).errs)

            champion_f = pop.champion_f.item()
            if champion_f < best:
                best = champion_f

        if i % 10 == 0:
            t = time.perf_counter() - t0
            print(f"t: {t:>3.0f}s, evolution {i:{width}}/{num_evolutions}, best: {best:5.0f}")

    evolutions = [[i] * pop_size * num_islands for i in range(1, num_evolutions + 1)]
    fs = np.concatenate(results["f"])
    xs = np.concatenate(results["x"])

    tof = np.cumsum(xs[:, : p.obj(xs[0]).number_of_legs + 1], axis=-1)
    tof = (tof[:, -1] - tof[:, 0]) / constants.JULIAN_YEAR

    df = pl.DataFrame(
        {
            "dv": fs[:, 0],
            "tof": tof,
            "x": xs,
            "gen": np.concatenate(evolutions),
        }
    )
    oglen = len(df)
    df = df.filter((pl.col("dv") != p.death_value))
    newlen = len(df)

    nerrs = pop.problem.extract(Problem).errs
    fevals = np.sum([x.get_population().problem.get_fevals() for x in archi])

    print(f"failed evaluations: {nerrs} out of {fevals}, or: {nerrs/fevals * 100:.0f}%")
    print(f"discarded: {oglen - newlen} out of {oglen}, or: {(oglen - newlen)/oglen * 100:.0f}%")

    return df, errs


df, errs = evolve(
    p,
    num_evolutions=400,
    num_generations=40,
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
selection = df.filter(pl.col("dv") <= 10_000)
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
plt.plot(errs[0] + np.diff(errs))

# %%
