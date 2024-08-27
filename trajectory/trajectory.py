# %%
from typing import Union, Any
from astropy.time import Time, TimeDelta
from functools import reduce
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pygmo as pg
import re

from tudatpy.trajectory_design.porkchop import porkchop, plot_porkchop
from tudatpy.trajectory_design.transfer_trajectory import TransferTrajectory
from tudatpy.trajectory_design import transfer_trajectory, shape_based_thrust
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy import constants

np.set_printoptions(precision=2, suppress=True)
sns.set_theme(style="ticks", palette="Set2")
pal = sns.color_palette("Set2")


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
            return []

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


# %%
central_body = "Sun"

transfer_body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = 3.8913e9
arrival_eccentricity = 0.999486

transfer_leg_settings, transfer_node_settings = (
    # transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
    #     transfer_body_order,
    #     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    # )
    # transfer_trajectory.mga_settings_dsm_velocity_based_legs(
    #     transfer_body_order,
    #     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    # )
    # transfer_trajectory.mga_settings_hodographic_shaping_legs_with_recommended_functions(
    #     transfer_body_order,
    #     [10, 20, 30],
    #     [1, 2, 3],
    #     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    # )
    transfer_trajectory.mga_settings_spherical_shaping_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
)

transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

bodies = environment_setup.create_simplified_system_of_bodies()

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body,
)

# %%
# What I want:
# - Pass custom bounds
#   - Omit bounds by passing "None" (using default)
# - Allow fixing of certain parameters
# - Allow Problem to be (cloud)pickled; thread-safe -> this is not possible atm :( (pybind11 allows pickling, not in tudatpy yet)
# - Function for running/plotting that handles dim=1,2 correctly without change


class Problem:
    def __init__(
        self,
        transfer_trajectory: TransferTrajectory,
        bounds: dict[str, dict] = {},
        fixed: dict[str, dict] = {},
        dim=1,
    ) -> None:
        self.transfer_trajectory = lambda: transfer_trajectory
        self.nn = transfer_trajectory.number_of_nodes
        self.nl = transfer_trajectory.number_of_legs
        self.fixed = dictify(fixed, leavelast=False)
        self.dim = dim

        self._determine_parameters()
        self._set_bounds(bounds)

    def get_nobj(self) -> int:
        return self.dim

    def get_bounds(self) -> tuple[list[float]]:
        bounds = np.array([*filter(lambda x: x is not None and len(x) > 0, self._get_bounds())])
        return bounds[:, 0], bounds[:, 1]

    def get_number_of_parameters(self) -> int:
        return len(self.get_bounds()[0])

    def fitness(self, dv: np.ndarray) -> np.ndarray:
        node_times, leg_parameters, node_parameters = self._convert_parameters(dv)

        try:
            self.evaluate(node_times, leg_parameters, node_parameters)
            delta_v = self.transfer_trajectory().delta_v
        except Exception:
            return self._death_penalty()

        if self.dim == 1:
            return [delta_v]

        tof = (node_times[-1] - node_times[0]) / constants.JULIAN_YEAR
        return [delta_v, tof]

    def evaluate(self, *args, **kwargs):
        if len(args) == 1:
            args = self._convert_parameters(*args)

        return self.transfer_trajectory().evaluate(*args, **kwargs)

    @property
    def death_value(self) -> float:
        return 1e16

    def _death_penalty(self):
        return [self.death_value] * self.dim

    def _convert_parameters(self, dv: np.ndarray) -> np.ndarray:
        bounds = self._get_bounds()

        parameters = [
            [] if (isinstance(x, (list, np.ndarray)) and len(x) == 0) else None for x in bounds
        ]

        def check(x):
            if x is None:
                return False

            if isinstance(x, (list, np.ndarray)):
                return len(x) > 0

            return True

        for i, x in zip(
            [i for i, x in enumerate(bounds) if check(x)],
            dv,
        ):
            parameters[i] = x

        fixed = []
        for k, v in sorted(
            self.fixed.items(),
            key=lambda x: ["departure", "leg_tof", "leg_parameters", "node_parameters"].index(x[0]),
        ):
            if not isinstance(v, dict):
                fixed.append(v)
                continue

            for k2, v2 in sorted(v.items(), key=lambda x: x[0]):
                if not isinstance(v2, dict):
                    fixed.append(v2)
                    continue

                for k3, v3 in sorted(v2.items(), key=lambda x: x[0]):
                    fixed.append(v3)

        for i, x in zip([i for i, x in enumerate(bounds) if x is None], fixed):
            parameters[i] = x

        nlp = sum([x[2] for x in self.legs])

        node_times = np.cumsum(parameters[: self.nl + 1])
        leg_parameters = []
        node_parameters = []

        lpar = parameters[self.nl + 1 : self.nl + 1 + nlp]
        npar = parameters[self.nl + 1 + nlp :]

        j = 0
        for i, x in enumerate(self.legs):
            leg_parameters.append(lpar[j : j + x[2]])
            j += x[2]

        j = 0
        for i, x in enumerate(self.nodes):
            node_parameters.append(npar[j : j + x[2]])
            j += x[2]

        return node_times, leg_parameters, node_parameters

    def _determine_parameters(self):
        nodes = [("node", None, 0)] * self.nn
        legs = [("leg", None, 0)] * self.nl

        node_times = np.linspace(1, 1e3, num=self.nn)
        leg_parameters = [[] for _ in range(self.nl)]
        node_parameters = [[] for _ in range(self.nn)]

        def try_evaluate(*parameters):
            try:
                self.evaluate(*parameters)
            except RuntimeError as e:
                match = re.match(
                    r"Error when getting (?:leg|node) parameters for (leg|node) ([0-9]*) \(type: (.*)\).*\. (?:Leg|Node) should have ([0-9]*) free parameters",
                    str(e),
                )

                if not match:
                    raise Exception("Error parsing error message:", e)

                sort, idx, _type, npar = match.groups()

                if sort == "leg":
                    arr = parameters[1]
                    kind = legs
                if sort == "node":
                    arr = parameters[2]
                    kind = nodes

                kind[int(idx)] = (sort, _type, int(npar))
                arr[int(idx)].extend([0.5 for _ in range(int(npar) - len(arr[int(idx)]))])

                return try_evaluate(*parameters)

        try_evaluate(node_times, leg_parameters, node_parameters)
        self.nodes = nodes
        self.legs = legs

    def _get_bounds(self):
        def chunk(acc: list, val: Union[list, np.ndarray, int, float]):
            if len(acc) > 0 and acc[-1] is not None and not isinstance(acc[-1], (list, np.ndarray)):
                acc.append([acc.pop(), val])
            else:
                acc.append(val)

            return acc

        return reduce(
            chunk,
            flatten(
                [
                    *undictify(self.bounds["departure"]),
                    *undictify(self.bounds["leg_tof"]),
                    *undictify(self.bounds["leg_parameters"]),
                    *undictify(self.bounds["node_parameters"]),
                ]
            ),
            [],
        )

    def _set_bounds(self, bounds: dict[str, dict]):
        updates = {}

        if "departure" in bounds:
            updates["departure"] = dictify(np.reshape(bounds["departure"], (1, 2)))

        if "leg_tof" in bounds:
            updates["leg_tof"] = dictify(bounds["leg_tof"])

        if "leg_parameters" in bounds:
            updates["leg_parameters"] = dictify(bounds["leg_parameters"])

        if "node_parameters" in bounds:
            updates["node_parameters"] = dictify(bounds["node_parameters"])

        bound_map = {
            "tof_fraction": [0, 1],
            "velocity_magnitude": [0, 1e6],
            "in-plane_angle": [0, 2 * np.pi],
            "out-of-plane_angle": [-np.pi / 2, np.pi / 2],
            "periapsis": [0, 1e26],
            "delta_v": [0, 10_000],
        }

        kind_map = {
            "leg": {
                "velocity-based DSM": ["tof_fraction"],
            },
            "node": {
                "escape_and_departure": [
                    "velocity_magnitude",
                    "in-plane_angle",
                    "out-of-plane_angle",
                ],
                "swingby": [
                    "periapsis",
                    "in-plane_angle",
                    "delta_v",
                ],
            },
        }

        def map_param(x):
            # An empty list signifies a 0 parameter leg/node for tudatpy
            if x[1] is None and x[2] == 0:
                return []

            if x[1] not in kind_map[x[0]]:
                raise Exception(f"Unknown {x[0]} kind: {x[1]}, no bounds found")

            return np.squeeze([bound_map[y] for y in kind_map[x[0]][x[1]]]).tolist()

        defaults = dict(
            leg_tof=dictify([[0, 5 * constants.JULIAN_YEAR] for _ in range(self.nl)]),
            leg_parameters=dictify([*map(map_param, self.legs)]),
            node_parameters=dictify([*map(map_param, self.nodes)]),
        )

        self.bounds = merge(defaults, updates)

        # Bounds that are for parameters that we have fixed are set to None
        for k, v in self.fixed.items():
            if not isinstance(v, dict):
                self.bounds[k] = None
                continue

            for num, v2 in v.items():
                if not isinstance(v2, dict):
                    self.bounds[k][num] = None
                    continue

                for i in v2.keys():
                    self.bounds[k][num][i] = None

        assert "departure" in self.bounds, "Departure bounds must be included"


node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]

departure = [[DateTime(2028, 4, 6).epoch(), DateTime(2033, 12, 31).epoch()]]
leg_tof = np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]]

bounds = dict(
    departure=departure,
    leg_tof=leg_tof,
)

fixed = dict(
    node_parameters={
        0: {0: 0},
        1: {2: 0},
        2: {2: 0},
    }
)

p = Problem(
    transfer_trajectory_object,
    bounds=bounds,
    fixed=fixed,
    dim=1,
)


def evolve(
    p: Problem,
    seed=4444,
    pop_size=40,
    num_generations=20,
    num_evolutions=600,
    **kwargs,
):
    prob = pg.problem(p)
    prob.c_tol = 1e-1

    pop = pg.population(prob, size=pop_size, seed=seed)

    algo_kwargs = dict(gen=num_generations, seed=seed, **kwargs)
    algo = pg.algorithm(pg.moead(**algo_kwargs) if p.dim == 2 else pg.sade(**algo_kwargs))

    results = dict(f=[pop.get_f()], x=[pop.get_x()])

    for i in range(1, num_generations + 1):
        pop = algo.evolve(pop)

        results["f"].append(pop.get_f())
        results["x"].append(pop.get_x())

        if i % 5 == 0:
            print(f"Generation {i}/{num_generations}...")

    generations = [[i] * pop_size for i in range(num_generations + 1)]
    fs = np.concatenate(results["f"])
    xs = np.concatenate(results["x"])

    tof = np.cumsum(xs[:, : p.nl + 1], axis=-1)
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

    print(f"num discarded: {oglen - newlen}, or: {(oglen - newlen)/oglen * 100:.0f}%")

    return df


df = evolve(
    p,
    num_generations=20,
    pop_size=40,
)

champions = (
    (df.group_by("gen").agg(pl.all().sort_by("dv").head(5)).explode(pl.all().exclude("gen")))
    .sort("dv")
    .filter(pl.col("gen") > 3)
)

plt.scatter(champions["tof"], champions["dv"], c=champions["gen"], cmap="viridis_r")

# %%
selection = df.filter(pl.col("dv") <= 30_000)

x = selection["x"].to_numpy()
J2000 = Time("J2000", format="jyear_str")

departures = J2000 + TimeDelta(x[:, 0], format="sec")
arrivals = J2000 + TimeDelta(np.sum(x[:, : p.nl + 1], axis=-1), format="sec")

dvs = selection["dv"]

plt.scatter(
    departures.to_value("datetime64"),
    arrivals.to_value("datetime64"),
    c=dvs,
    cmap="viridis_r",
)
plt.colorbar(label=r"$\Delta$V")
plt.xticks(rotation=-45)
plt.xlabel("Departure date")
plt.ylabel("Arrival date")

# %%
p.evaluate(df.sort("dv")[0]["x"].item().to_numpy())
t = p.transfer_trajectory()
t.delta_v_per_leg, t.delta_v_per_node

# %%


# %%
# Types of trajectory:
# 1. Continuous low-thrust (Ion, SEP) (can be non-JGA)
# 2. Impulsive (chemical) (can be non-JGA)
# 3. Unpowered (has to be JGA)
# -> can all do JGA? Feasible with/without?

# Tudatpy:
# 1. Unpowered unperturbed: dv at periapsis of swingby node
# 2. DSM velocity-based: ^, also at tof during leg


# calculate_lambert_arc_impulsive_delta_v
# https://github.com/tudat-team/tudatpy/blob/60719404b37c8cfb747e9d4cd7ef6f865b148acf/tudatpy/trajectory_design/porkchop/_lambert.py#L20
def delta_v_function(
    bodies,
    departure_body,
    target_body,
    departure_epoch,
    arrival_epoch,
    central_body="Sun",
) -> tuple[float, float]:
    return [10, 20]


porkchop()

# %%
#
[
    ["leg", 0, "velocity-based DSM", "tof-fraction", 0],
    ["node", 0, "escape_and_departure", "velocity_magnitude", 0],
]

bound_map = {
    "tof_fraction": {"type": "float", "bound": [0, 1]},
}

kind_map = {
    "leg": {
        "velocity-based DSM": ["tof_fraction"],
    },
    "node": {
        "escape_and_departure": [
            "velocity_magnitude",
            "in-plane_angle",
            "out-of-plane_angle",
        ],
        "swingby": [
            "periapsis",
            "in-plane_angle",
            "delta_v",
        ],
    },
}


parameters = []

for i in range(p.nn):
    parameters.append(["time", i, "time", "time", 0])

for kind in [p.legs, p.nodes]:
    for i, x in enumerate(kind):
        for j in range(x[2]):
            parameters.append([x[0], i, x[1], kind_map[x[0]][x[1]][j], j])

parameters

# %%
# - see if creation of object requires any parameters (for creation)
# - evaluate to build up list of required parameters (for `evaluate`)
# - maintain index for any of these params which:
#       - are fixed
#       - have a transform
# - form bounds
# - `fitness` will give vector of creation + evaluate

# creation
#   *args, **kwargs
# parameters
#   node_times, leg_parameters, node_parameters

# some nodes, legs have no parameters

[10, 50, 900, 80]

[
    ["node", 0, "velocity-based DSM", "tof_fraction", 0],
    None,  # disabled
    ["leg", "..."],
]

# full info
{
    "creation": dict(args=[], kwargs={}),
    "parameters": {
        "node_times": [],
        "leg_parameters": [
            [],  # no parameters
            ["tof_fraction"],
            [],
        ],
        "node_parameters": [
            [
                "periapsis",
            ]
        ],
    },
}

p.nodes

# %%
bound_map = {
    "tof_fraction": [0, 1],
}

type_map = {
    "tof_fraction": "float",
}

parameter_map = {
    "leg": {
        "velocity-based DSM": ["tof_fraction"],
    },
    "node": {
        "escape_and_departure": [
            "velocity_magnitude",
            "in-plane_angle",
            "out-of-plane_angle",
        ],
        "swingby": [
            "periapsis",
            "in-plane_angle",
            "delta_v",
        ],
    },
}

# %%
transfer_trajectory_object
# %%
