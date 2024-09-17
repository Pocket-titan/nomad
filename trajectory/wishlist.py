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


FOLDER = Path(__file__).parent / "runs"


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
        ("Earth", "Neptune"): "long",
        ("Earth", "Jupiter"): "long",
        ("Earth", "Saturn"): "long",
        ("Earth", "Mars"): "short",
        ("Earth", "Venus"): "short",
        ("Mars", "Venus"): "short",
        ("Mars", "Jupiter"): "long",
        ("Mars", "Saturn"): "long",
        ("Mars", "Neptune"): "long",
        ("Venus", "Jupiter"): "long",
        ("Venus", "Saturn"): "long",
        ("Venus", "Neptune"): "long",
        ("Jupiter", "Neptune"): "long",
        ("Jupiter", "Saturn"): "long",
        ("Saturn", "Neptune"): "long",
    }

    if (destination, target) not in bound_map:
        if (target, destination) in bound_map:
            destination, target = target, destination
        else:
            destination, target = "_", "_"

    return [
        TimeDelta(x, format="jd").to_value("sec")
        for x in time_map[bound_map[(destination, target)]]
    ]


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
    if args.dry:
        print("Dry run, not writing to file")

    if args.overwrite or not (FOLDER / "wishlist.pkl").exists():
        print("Creating new wishlist")
        prev = []
    else:
        with open(FOLDER / "wishlist.pkl", "rb") as f:
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
                suffix = f"_{preset}"
                suffix += presets[preset].get("suffix", "")
                wishlist.append({**presets[preset], "suffix": suffix})

            if isinstance(presets[preset], list):
                print(f"Adding preset: {preset} with {len(presets[preset])} options")
                wishlist.extend(presets[preset])
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
        with open(FOLDER / "wishlist.pkl", "wb") as f:
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


# algo_map = {
#     "gaco": lambda: pg.gaco(
#         gen=1000,
#         ker=15,
#         q=1.0,
#         oracle=1e9,
#         acc=0.01,
#         threshold=500,
#         n_gen_mark=7,
#         seed=4444,
#     ),
#     "pso": lambda: pg.pso(gen=1000, seed=4444),
#     "sga": lambda: pg.sga(gen=1000, seed=4444),
#     "de": lambda: pg.de(gen=1000, F=0.5, seed=4444),
#     "sade": lambda: pg.sade(gen=1000, seed=4444),
#     "slsqp": lambda: pg.nlopt("slsqp"),
#     # "snopt7": snopt7_algo,
# }


# algos = {
#     "gaco": pg.gaco(
#         gen=100,
#         ker=15,
#         q=1.0,
#         oracle=1e9,
#         acc=0.01,
#         threshold=50,
#         n_gen_mark=7,
#         seed=4444,
#     ),
#     "pso": pg.pso(gen=100, seed=4444),
#     "sga": pg.sga(gen=100, seed=4444),
#     "de": pg.de(gen=100, F=0.5, seed=4444),
#     "sade": pg.sade(gen=100, seed=4444),
#     "slsqp": pg.nlopt("slsqp"),
#     "sa": pg.simulated_annealing(Ts=10.0, Tf=0.1, n_T_adj=10, seed=4444),
# }

algo_kwargs = {
    "gaco": dict(
        gen=100,
        ker=15,
        q=1.0,
        oracle=1e9,
        acc=0.01,
        threshold=50,
        n_gen_mark=7,
        seed=4444,
    ),
    "pso": dict(gen=100, seed=4444),
    "sga": dict(gen=100, seed=4444),
    "de": dict(gen=100, F=0.5, seed=4444),
    "sade": dict(gen=100, seed=4444),
    "slsqp": dict(solver="slsqp"),
    "sa": dict(Ts=10.0, Tf=0.1, n_T_adj=10, seed=4444),
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
    "low": dict(
        num_evolutions=50,
        num_generations=25,
        pop_size=10,
    ),
    "medium": dict(
        num_evolutions=300,
        num_generations=100,
        pop_size=200,
    ),
    "high": dict(
        num_evolutions=600,
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

departure = [
    DateTime(start.year, start.month, start.day, start.hour).epoch(),
    DateTime(end.year, end.month, end.day, end.hour).epoch(),
]

leg_tof = [
    [TimeDelta(y, format="jd").to_value("sec") for y in x]
    for x in [
        [30, 400],
        [100, 470],
        [30, 400],
        [400, 2000],
        [1000, 6000],
    ]
]

cassini_bounds = dict(departure=departure, leg_tof=leg_tof)


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
        [dict(evolve_kwargs=evolve_map["high"])],
        [
            dict(evolve_kwargs=dict(algo_name=k, algo_kwargs=v))
            for k, v in algo_kwargs.items()
        ],
    )
]

for x in cassini_bunch:
    key = x["evolve_kwargs"]["algo_name"]
    x["suffix"] = f"_cassini_{key}"


# %%
presets = dict(cassini=cassini, cassini_bunch=cassini_bunch)


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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
