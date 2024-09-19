# %%
from argparse import ArgumentParser
from frozendict import frozendict
from tudatpy.astro.time_conversion import DateTime
import pygmo_plugins_nonfree as ppnf
from astropy.time import Time, TimeDelta
from pydash import clone_deep, merge
from itertools import product
from functools import wraps
from pprint import pprint
from pathlib import Path
import cloudpickle as pkl
import pygmo as pg
import numpy as np


FOLDER = Path(__file__).parent


# create_obj
def create_unpowered(
    *pars,
    body_order=None,
    departure_semi_major_axis=np.inf,
    departure_eccentricity=0.0,
    arrival_semi_major_axis=3.8913e9,
    arrival_eccentricity=0.999486,
    minimum_pericenters=None,
):
    from tudatpy.numerical_simulation.environment_setup import (
        create_simplified_system_of_bodies,
    )
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        mga_settings_unpowered_unperturbed_legs,
    )

    central_body = "Sun"

    bodies = create_simplified_system_of_bodies()

    leg_settings, node_settings = mga_settings_unpowered_unperturbed_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
        **(
            dict(minimum_pericenters=minimum_pericenters)
            if minimum_pericenters is not None
            else {}
        ),
    )

    obj = create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )

    return obj, leg_settings, node_settings


def create_dsm_velocity(
    *pars,
    body_order=None,
    departure_semi_major_axis=np.inf,
    departure_eccentricity=0.0,
    arrival_semi_major_axis=3.8913e9,
    arrival_eccentricity=0.999486,
    minimum_pericenters=None,
):
    from tudatpy.numerical_simulation.environment_setup import (
        create_simplified_system_of_bodies,
    )
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        mga_settings_dsm_velocity_based_legs,
    )

    central_body = "Sun"

    bodies = create_simplified_system_of_bodies()

    leg_settings, node_settings = mga_settings_dsm_velocity_based_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
        **(
            dict(minimum_pericenters=minimum_pericenters)
            if minimum_pericenters is not None
            else {}
        ),
    )

    obj = create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )

    return obj, leg_settings, node_settings


def create_1dsm(
    *pars,
    body_order=None,
    departure_semi_major_axis=np.inf,
    departure_eccentricity=0.0,
    arrival_semi_major_axis=3.8913e9,
    arrival_eccentricity=0.999486,
    minimum_pericenters={
        "Earth": 6678000,
        "Jupiter": 600000000,
        "Mars": 3689000,
        "Mercury": 2740000,
        "Saturn": 70000000,
        "Venus": 6351800,
    },
    dsm_leg_index=None,
):
    from tudatpy.numerical_simulation.environment_setup import (
        create_simplified_system_of_bodies,
    )
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        departure_node,
        swingby_node,
        capture_node,
        unpowered_leg,
        dsm_velocity_based_leg,
    )

    central_body = "Sun"

    bodies = create_simplified_system_of_bodies()

    node_settings = [
        departure_node(departure_semi_major_axis, departure_eccentricity),
        *[swingby_node(minimum_pericenters[x]) for x in body_order[1:-1]],
        capture_node(arrival_semi_major_axis, arrival_eccentricity),
    ]

    leg_settings = [
        unpowered_leg() if i != dsm_leg_index else dsm_velocity_based_leg()
        for i in range(len(body_order) - 1)
    ]

    obj = create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )

    return obj, leg_settings, node_settings


create_objs = [create_unpowered, create_dsm_velocity]

defaults = dict(
    body_order=None,
    create_obj=None,
    p_kwargs=dict(
        bounds=dict(
            departure=[
                DateTime(2030, 1, 1).epoch(),
                DateTime(2045, 1, 1).epoch(),
            ],
        ),
        fixed=dict(),
        dim=1,
    ),
    evolve_kwargs=dict(
        num_evolutions=250,
        num_generations=100,
        num_islands=None,
        pop_size=100,
        seed=4444,
    ),
    extra=dict(),
)


def with_kwargs(fn, **kwargs):
    @wraps(fn)
    def wrapper(*args, **_kwargs):
        return fn(*args, **_kwargs, **kwargs)

    return wrapper


def get_leg_tof_bounds(destination, target):
    time_map = {
        "short": [30, 365 * 1],
        "medium": [30 * 6, 365 * 3],
        "long": [365 * 5, 365 * 15],
    }  # in days

    bound_map = {
        ("_", "_"): "medium",
        ("Earth", "Neptune"): [365 * 12, 365 * 20],
        ("Earth", "Jupiter"): [365 * 1, 365 * 6],
        ("Earth", "Saturn"): [365 * 6.5, 365 * 19],
        ("Earth", "Mars"): [365 * 0.5, 365 * 2],
        ("Earth", "Venus"): [365 * 0.1, 365 * 1.5],
        ("Mars", "Venus"): [365 * 0.3, 365 * 1.5],
        ("Mars", "Jupiter"): [365 * 1.5, 365 * 4.5],
        ("Mars", "Saturn"): [365 * 5.5, 365 * 12],
        ("Mars", "Neptune"): [365 * 11, 365 * 20],
        ("Venus", "Jupiter"): [365 * 2.5, 365 * 6],
        ("Venus", "Saturn"): [365 * 6, 365 * 14],
        ("Venus", "Neptune"): [365 * 12, 365 * 17],
        ("Jupiter", "Neptune"): [365 * 10, 365 * 19],
        ("Jupiter", "Saturn"): [365 * 2, 365 * 16],
        ("Saturn", "Neptune"): [365 * 7, 365 * 15],
        ("Venus", "Venus"): [365 * 0.2, 365 * 1.5],
        ("Earth", "Earth"): [365 * 0.2, 365 * 1.5],
    }

    if (destination, target) not in bound_map:
        if (target, destination) in bound_map:
            destination, target = target, destination
        else:
            print("WARNING: No bounds found for", destination, target)
            destination, target = "_", "_"

    bounds = [
        TimeDelta(x, format="jd").to_value("sec")
        for x in bound_map[(destination, target)]
    ]

    return [time_map[x] if x in time_map else x for x in bounds]


def generate_combinations(
    create_obj,
    body_order,
    evolve_kwargs=None,
    p_kwargs=None,
    extra=None,
    suffix=None,
):
    if evolve_kwargs is None:
        evolve_kwargs = [defaults["evolve_kwargs"]]

    if p_kwargs is None:
        p_kwargs = [defaults["p_kwargs"]]

    if extra is None:
        extra = [defaults["extra"]]

    wishlist = [
        {
            "body_order": x[0],
            "create_obj": x[1],
            "p_kwargs": x[2],
            "evolve_kwargs": x[3],
            "extra": x[4],
        }
        for x in product(
            body_order,
            create_obj,
            p_kwargs,
            evolve_kwargs,
            extra,
        )
    ]

    return wishlist


def generate_suffix(x):
    return f"_{x['create_obj'].__name__.replace('create_', '')}"


def main(args):
    folder = FOLDER / args.folder
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    if args.dry:
        print("Dry run, not writing to file")

    if args.overwrite or not (folder / "wishlist.pkl").exists():
        print("Creating new wishlist")
        prev = []
    else:
        with open(folder / "wishlist.pkl", "rb") as f:
            prev = pkl.load(f)
            print(f"Loaded wishlist with {len(prev)} entries, adding new entries to it")

    wishlist = [*prev]

    if args.preset:
        for preset in args.preset:
            if preset not in presets:
                print(f"Unknown preset: {preset}")
                continue

            if isinstance(presets[preset], dict):
                print(f"Adding preset: {preset} with options:")
                pprint(presets[preset], compact=True, width=70)
                # suffix = f"_{preset}"
                # suffix += presets[preset].get("suffix", "")
                wishlist.append({**presets[preset]})

            if isinstance(presets[preset], list):
                print(f"Adding preset: {preset} with {len(presets[preset])} options")
                new = []

                for x in presets[preset]:
                    # suffix = f"_{preset}"
                    # suffix += x.get("suffix", "")
                    new.append({**x})

                wishlist.extend(new)
    else:
        pars = [
            args.create_obj,
            args.body_order,
            args.evolve_kwargs,
            args.p_kwargs,
            args.extra,
        ]
        pars = [x for x in pars if x is not None]
        print("Making linear combinations with args:")
        print(f"  {' x '.join([str(x) for x in pars])}")
        wishlist.extend(generate_combinations(*pars))

    wishlist = [merge(clone_deep(defaults), x) for x in wishlist]

    def filter_wishlist(x):
        return True

    def map_wishlist(x):
        if "leg_tof" not in x["p_kwargs"]["bounds"]:
            num_legs = len(x["body_order"]) - 1
            x["p_kwargs"]["bounds"]["leg_tof"] = [
                get_leg_tof_bounds(x["body_order"][i - 1], x["body_order"][i])
                for i in range(1, num_legs + 1)
            ]

        x["create_obj"] = with_kwargs(
            x["create_obj"],
            body_order=x["body_order"],
            **(x["extra"] if "extra" in x else {}),
        )

        return x

    wishlist = [map_wishlist(x) for x in wishlist if filter_wishlist(x)]

    for i, x in enumerate(wishlist):
        if "suffix" not in x:
            x["suffix"] = generate_suffix(x)
        else:
            x["suffix"] = f"{x['suffix']}{generate_suffix(x)}"

    if len(wishlist) > 0 and not args.dry:
        print(f"Generated wishlist with {len(wishlist)} entries")
        with open(folder / "wishlist.pkl", "wb") as f:
            pkl.dump(wishlist, f)


# Arguments
body_map = {
    "E": "Earth",
    "M": "Mars",
    "J": "Jupiter",
    "N": "Neptune",
    "V": "Venus",
    "S": "Saturn",
}

create_map = {
    "unpowered": create_unpowered,
    "dsm_velocity": create_dsm_velocity,
}


def snopt7_algo():
    snopt7 = ppnf.snopt7()
    snopt7.set_integer_option("Major iterations limit", 1000)
    snopt7.set_integer_option("Iterations limit", 200000)
    snopt7.set_numeric_option("Major optimality tolerance", 1e-2)
    snopt7.set_numeric_option("Major feasibility tolerance", 1e-8)
    return snopt7


algo_kwargs = {
    "gaco": dict(
        gen=500,
        ker=15,
        q=1.0,
        oracle=1e9,
        acc=0.01,
        threshold=250,
        n_gen_mark=7,
        seed=4444,
    ),
    "pso": dict(gen=500, seed=4444),
    "sga": dict(gen=500, seed=4444),
    "de": dict(gen=500, F=0.5, seed=4444),
    "sade": dict(gen=500, seed=4444),
    # "slsqp": dict(solver="slsqp"),
    "sa": dict(Ts=10.0, Tf=0.1, n_T_adj=10, seed=4444),
    # new
    "de1220": dict(gen=500, seed=4444),
    "cmaes": dict(gen=500, seed=4444),
    "ihs": dict(gen=500, seed=4444),
    "nsga2": dict(gen=500, seed=4444),
}


for key in list(algo_kwargs.keys()):
    algo_kwargs[f"mbh_{key}"] = dict(
        mbh=dict(stop=5, perturb=0.05, seed=4444),
        algo=algo_kwargs[key],
    )

algo_order = [
    "de",
    "sade",
    "gaco",
    "pso",
    "sga",
    "sa",
    "slsqp",
    "mbh_de",
    "mbh_sade",
    "mbh_gaco",
    "mbh_pso",
    "mbh_sga",
    "mbh_sa",
    "mbh_slsqp",
]

algo_names = list(algo_kwargs.keys())

evolve_map = {
    "test": dict(
        num_evolutions=2,
        num_generations=1,
        pop_size=5,
    ),
    "min": dict(
        num_evolutions=10,
        num_generations=100,
        pop_size=25,
    ),
    "low": dict(
        num_evolutions=50,
        num_generations=25,
        pop_size=100,
    ),
    "medium": dict(
        num_evolutions=100,
        num_generations=100,
        pop_size=200,
    ),
    "high": dict(
        num_evolutions=200,
        num_generations=250,
        pop_size=300,
    ),
    "ultra": dict(
        num_evolutions=1000,
        num_generations=1000,
        pop_size=4000,
    ),
}

# Presets
# Cassini MGA problem
MJD2000 = Time(2000, format="jyear").to_value("mjd")
start = Time(MJD2000 - 1000, format="mjd").to_value("datetime")
end = Time(MJD2000 - 0, format="mjd").to_value("datetime")

cassini_departure = [
    DateTime(start.year, start.month, start.day, start.hour).epoch(),
    DateTime(end.year, end.month, end.day, end.hour).epoch(),
]

cassini_leg_tof = [
    [TimeDelta(y, format="jd").to_value("sec") for y in x]
    for x in [
        [30, 400],
        [100, 470],
        [30, 400],
        [400, 2000],
        [1000, 6000],
    ]
]

[
    [a / 365, b / 365]
    for a, b in [
        [30, 400],
        [100, 470],
        [30, 400],
        [400, 2000],
        [1000, 6000],
    ]
]

cassini_bounds = dict(departure=cassini_departure, leg_tof=cassini_leg_tof)


cassini = {
    "body_order": ["Earth", "Venus", "Venus", "Earth", "Jupiter", "Saturn"],
    "create_obj": create_unpowered,
    "p_kwargs": dict(bounds=cassini_bounds, fixed=dict(), dim=1),
    "evolve_kwargs": {
        "num_generations": 1000,
        "num_evolutions": 1000,
        "pop_size": 20,
        "algo_name": "pso",
    },
    "extra": dict(
        arrival_semi_major_axis=108950e3 / 0.02,
        arrival_eccentricity=0.98,
        # arrival_semi_major_axis=np.inf,
        # arrival_eccentricity=0,
        minimum_pericenters={
            "Venus": 6351.8e3,
            "Earth": 6778.1e3,
            "Jupiter": 600000e3,
        },
    ),
}

cassini_bunch = [
    merge(*clone_deep(x))
    for x in product(
        [cassini],
        [dict(evolve_kwargs=evolve_map["min"])],
        [
            dict(evolve_kwargs=dict(algo_name=k, algo_kwargs=v))
            for k, v in algo_kwargs.items()
        ],
        [dict(evolve_kwargs=dict(num_islands=12))],
    )
]

cassini_wide_departure = [
    DateTime(1997, 1, 1).epoch(),
    DateTime(2000, 1, 1).epoch(),
]

cassini_wide = [
    merge(*clone_deep(x))
    for x in product(
        [cassini],
        [
            dict(
                p_kwargs=dict(
                    bounds=dict(
                        departure=cassini_wide_departure,
                    )
                )
            )
        ],
        [
            dict(
                evolve_kwargs={
                    "num_evolutions": 50,
                    # "num_evolutions": 200,
                    # "num_generations": 50,
                    "num_generations": 25,
                    "pop_size": 100,
                    # "pop_size": 100,
                    # "seed": 4444,
                    "seed": 19000,
                }
            )
        ],
        [
            dict(evolve_kwargs=dict(algo_name=k, algo_kwargs=algo_kwargs[k]))
            for k in [
                # "gaco",
                # "mbh_gaco",
                # "pso",
                # "mbh_pso",
                "sade",
                # "mbh_sade",
            ]
        ],
        [
            {
                "extra": dict(
                    # arrival_semi_major_axis=np.inf,
                    # arrival_eccentricity=0,
                    # arrival_semi_major_axis=np.inf,
                    # arrival_eccentricity=0,
                    minimum_pericenters={
                        "Venus": 6351.8e3,
                        "Earth": 6778.1e3,
                        "Jupiter": 600000e3,
                    },
                ),
            }
        ],
    )
]


# Neptune presets
# EN, EJN (odyssey)
# EJN, EMJN (triton oceon world surveyor)
# EVEEJN (trident discovery mission)
# ESN (neptune mission analysis pygmo paper, 1dsm only)
# EN (1dsm only)

neptune_body_orders = [
    # ["Earth", "Neptune"],
    ["Earth", "Jupiter", "Neptune"],
    ["Earth", "Mars", "Jupiter", "Neptune"],
    # ["Earth", "Venus", "Earth", "Jupiter", "Neptune"],
    ["Earth", "Venus", "Earth", "Earth", "Jupiter", "Neptune"],
]

neptune_yearly_bounds = [
    dict(
        departure=[
            DateTime(2030 + i, 1, 1).epoch(),
            DateTime(2030 + i + 3, 12, 31).epoch(),
        ]
    )
    for i in range(0, 15, 3)
]


neptune_bounds = dict(
    departure=[DateTime(2030, 1, 1).epoch(), DateTime(2045, 1, 1).epoch()],
)

neptune_extra = dict(
    departure_semi_major_axis=np.inf,
    departure_eccentricity=0.0,
    # arrival_semi_major_axis=np.inf,
    # arrival_eccentricity=0.0,
    arrival_semi_major_axis=172320e3 / 0.02,  # roughly 10% of SOI
    arrival_eccentricity=0.98,
)

neptune = {
    "p_kwargs": dict(bounds=neptune_bounds, fixed=dict(), dim=1),
    "extra": neptune_extra,
}

neptune_unpowered_body_orders = [
    ["Earth", "Jupiter", "Neptune"],
    ["Earth", "Mars", "Jupiter", "Neptune"],
    ["Earth", "Venus", "Earth", "Earth", "Jupiter", "Neptune"],
]


neptune_unpowered = [
    merge(*clone_deep(x))
    for x in product(
        [neptune],
        [dict(create_obj=create_unpowered)],
        [dict(body_order=x) for x in neptune_unpowered_body_orders],
        [dict(p_kwargs=dict(bounds=x)) for x in neptune_yearly_bounds],
        [
            dict(
                evolve_kwargs={
                    "num_evolutions": 100,
                    "num_generations": 25,
                    "pop_size": 200,
                    "seed": 4444,
                }
            )
        ],
        [
            dict(evolve_kwargs=dict(algo_name=k, algo_kwargs=algo_kwargs[k]))
            for k in [
                "sade",
                "gaco",
                "mbh_gaco",
                "pso",
            ]
        ],
        [
            dict(
                extra=dict(
                    departure_semi_major_axis=np.inf,
                    departure_eccentricity=0.0,
                    arrival_semi_major_axis=np.inf,
                    arrival_eccentricity=0.0,
                    minimum_pericenters={
                        "Earth": 6371e3 * 1.5,
                        "Venus": 6051.8e3 * 1.5,
                        "Jupiter": 69_911e3 * 1.5,
                        "Saturn": 58_232e3 * 1.5,
                        "Mars": 3389.5e3 * 1.5,
                    },
                )
            )
        ],
    )
]

neptune_1dsm_body_orders = [
    ["Earth", "Neptune"],
    ["Earth", "Jupiter", "Neptune"],
]

neptune_1dsm = [
    merge(*clone_deep(x))
    for x in product(
        [neptune],
        [dict(create_obj=create_1dsm)],
        [dict(body_order=x) for x in neptune_1dsm_body_orders],
        [dict(p_kwargs=dict(bounds=x)) for x in neptune_yearly_bounds],
        [
            dict(
                evolve_kwargs={
                    "num_evolutions": 100,
                    "num_generations": 25,
                    "pop_size": 200,
                    "seed": 4444,
                }
            )
        ],
        [
            dict(evolve_kwargs=dict(algo_name=k, algo_kwargs=algo_kwargs[k]))
            for k in [
                "sade",
                "gaco",
                "mbh_gaco",
                "pso",
            ]
        ],
        [
            dict(
                extra=dict(
                    departure_semi_major_axis=np.inf,
                    departure_eccentricity=0.0,
                    arrival_semi_major_axis=np.inf,
                    arrival_eccentricity=0.0,
                    minimum_pericenters={
                        "Earth": 6371e3 * 1.5,
                        "Venus": 6051.8e3 * 1.5,
                        "Jupiter": 69_911e3 * 1.5,
                        "Saturn": 58_232e3 * 1.5,
                        "Mars": 3389.5e3 * 1.5,
                    },
                )
            )
        ],
        [dict(extra=dict(dsm_leg_index=0)), dict(extra=dict(dsm_leg_index=1))],
    )
]


def filter_1dsm(x):
    if "extra" in x and "dsm_leg_index" in x["extra"]:
        if x["extra"]["dsm_leg_index"] == 1 and len(x["body_order"]) == 2:
            return False

    return True


def map_1dsm(x):
    J2000 = Time(2000, format="jyear")
    dep_year = J2000 + TimeDelta(x["p_kwargs"]["bounds"]["departure"][0], format="sec")
    x["suffix"] = (
        f"_1dsm_{x['extra']['dsm_leg_index']}_{float(str(dep_year)):.0f}_{x['evolve_kwargs']['algo_name']}"
    )
    return x


neptune_1dsm = [map_1dsm(x) for x in neptune_1dsm if filter_1dsm(x)]

groups = {}
for x in neptune_1dsm:
    key = tuple(x["p_kwargs"]["bounds"]["departure"])

    if key not in groups:
        groups[key] = []

    groups[key].append(x)

group_keys = list(groups.keys())
neptune_1dsm_0 = groups[group_keys[0]]
neptune_1dsm_1 = groups[group_keys[1]]
neptune_1dsm_2 = groups[group_keys[2]]
neptune_1dsm_3 = groups[group_keys[3]]
neptune_1dsm_4 = groups[group_keys[4]]


# uranus
uranus_body_orders = [
    # ["Earth", "uranus"],
    # ["Earth", "Jupiter", "uranus"],
    # ["Earth", "Mars", "Jupiter", "uranus"],
    # ["Earth", "Venus", "Earth", "Jupiter", "uranus"],
    ["Earth", "Earth", "Jupiter", "Uranus"]
]

uranus_node_times = [
    DateTime(2031, 6, 20).epoch(),
    DateTime(2033, 4, 27).epoch(),
    DateTime(2035, 12, 21).epoch(),
    DateTime(2044, 11, 5).epoch(),
]

uranus_leg_tof = np.diff(uranus_node_times)[:, None] * [[0.5, 2]]


uranus_bounds = dict(
    departure=[DateTime(2031, 1, 7).epoch(), DateTime(2031, 12, 3).epoch()],
    leg_tof=uranus_leg_tof,
)

uranus_extra = dict(
    departure_semi_major_axis=np.inf,
    departure_eccentricity=0.0,
    arrival_semi_major_axis=np.inf,
    arrival_eccentricity=0.0,
    # arrival_semi_major_axis=172320e3 / 0.02,  # roughly 10% of SOI
    # arrival_eccentricity=0.98,
)

uranus = {
    "p_kwargs": dict(bounds=uranus_bounds, fixed=dict(), dim=1),
    "extra": uranus_extra,
}


uranus_1dsm = [
    merge(*clone_deep(x))
    for x in product(
        [uranus],
        [dict(create_obj=create_1dsm)],
        [dict(body_order=x) for x in uranus_body_orders],
        [
            dict(
                evolve_kwargs={
                    **evolve_map["medium"],
                    "num_evolutions": 25,
                    "num_generations": 50,
                    "pop_size": 50,
                }
            )
        ],
        # [
        #     dict(evolve_kwargs=dict(algo_name=k, algo_kwargs=v))
        #     for k, v in algo_kwargs.items()
        # ],
        [
            dict(
                evolve_kwargs=dict(
                    algo_name="mbh_pso",
                    algo_kwargs=algo_kwargs["mbh_pso"],
                )
            )
        ],
        [dict(extra=dict(dsm_leg_index=0))],
    )
]


# %%
presets = dict(
    cassini=cassini,
    cassini_b=cassini_bunch,
    cassini_wide=cassini_wide,
    neptune_unpowered=neptune_unpowered,
    uranus=uranus_1dsm,
    neptune_1dsm_0=neptune_1dsm_0,
    neptune_1dsm_1=neptune_1dsm_1,
    neptune_1dsm_2=neptune_1dsm_2,
    neptune_1dsm_3=neptune_1dsm_3,
    neptune_1dsm_4=neptune_1dsm_4,
)

for k, v in presets.items():
    if isinstance(v, list):
        for x in v:
            if "suffix" not in x or x["suffix"] == "":
                key = x["evolve_kwargs"]["algo_name"]
                x["suffix"] = f"_{k}_{key}"

    if isinstance(v, dict):
        if "suffix" not in v or v["suffix"] == "":
            key = v["evolve_kwargs"]["algo_name"]
            v["suffix"] = f"_{k}_{key}"


def parse_args():
    def comma_arg(value):
        return value.split(",")

    parser = ArgumentParser(prog="wishlist")
    parser.add_argument(
        "create_obj",
        type=comma_arg,
        nargs="?",
        help="create_obj",
    )
    parser.add_argument(
        "body_order",
        type=comma_arg,
        nargs="?",
        help="body_order",
    )
    parser.add_argument(
        "evolve_kwargs",
        type=comma_arg,
        nargs="?",
        help="evolve_kwargs",
    )
    parser.add_argument(
        "p_kwargs",
        type=comma_arg,
        nargs="?",
        help="p_kwargs",
    )
    parser.add_argument(
        "extra",
        type=comma_arg,
        nargs="?",
        help="extra",
    )

    # kw args
    parser.add_argument(
        "-p",
        "--preset",
        type=comma_arg,
        help="Add config from existing preset(s)",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing wishlist",
    )
    parser.add_argument(
        "-d",
        "--dry",
        action="store_true",
        help="Dry run (do not write to file)",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="runs",
        help="Folder to put runs in",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
