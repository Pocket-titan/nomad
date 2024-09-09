# %%
from tudatpy.astro.time_conversion import DateTime
from pydash import clone_deep, merge
from astropy.time import TimeDelta
from itertools import product
from functools import wraps
from pathlib import Path
import cloudpickle as pkl
import pygmo as pg
import numpy as np
import sys
import os
from pprint import pprint


# body_order
body_orders = [
    ["Earth", "Neptune"],
    ["Earth", "Jupiter", "Neptune"],
    ["Earth", "Saturn", "Neptune"],
    ["Earth", "Mars", "Jupiter", "Neptune"],
    ["Earth", "Jupiter", "Saturn", "Neptune"],
    ["Earth", "Venus", "Earth", "Jupiter", "Neptune"],
    ["Earth", "Venus", "Earth", "Jupiter", "Saturn", "Neptune"],
    ["Earth", "Venus", "Earth", "Mars", "Jupiter", "Neptune"],
]


# create_obj
def create_unpowered(body_order, *pars):
    from tudatpy.numerical_simulation.environment_setup import (
        create_simplified_system_of_bodies,
    )
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        mga_settings_unpowered_unperturbed_legs,
    )

    central_body = "Sun"

    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = 3.8913e9
    arrival_eccentricity = 0.999486

    bodies = create_simplified_system_of_bodies()

    leg_settings, node_settings = mga_settings_unpowered_unperturbed_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )

    obj = create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )

    return obj, leg_settings, node_settings


def create_dsm_velocity(body_order, *pars):
    from tudatpy.numerical_simulation.environment_setup import (
        create_simplified_system_of_bodies,
    )
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        mga_settings_dsm_velocity_based_legs,
    )

    central_body = "Sun"

    departure_semi_major_axis = np.inf
    departure_eccentricity = 0.0

    arrival_semi_major_axis = 3.8913e9
    arrival_eccentricity = 0.999486

    bodies = create_simplified_system_of_bodies()

    leg_settings, node_settings = mga_settings_dsm_velocity_based_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
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

# p_kwargs
# node_times from Neptune Odyssey
node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]

simple_bounds = dict(
    departure=[DateTime(2030, 1, 1).epoch(), DateTime(2045, 1, 1).epoch()],
    # leg_tof=np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]],
)

simple_p_kwargs = dict(bounds=simple_bounds, fixed=dict())

p_kwargss = [simple_p_kwargs]

# evolve_kwargs
simple_evolve_kwargs = dict(
    # num_evolutions=500,
    # num_generations=250,
    # pop_size=100,
    num_evolutions=15,
    num_generations=10,
    pop_size=10,
    seed=4444,
    algo=lambda: pg.mbh(algo=pg.nlopt("slsqp"), stop=5, perturb=0.01, seed=4444),
)

evolve_kwargss = [simple_evolve_kwargs]

defaults = dict(
    body_order=None,
    create_obj=None,
    p_kwargs=dict(
        bounds=dict(),
        fixed=dict(),
        dim=1,
    ),
    evolve_kwargs=dict(
        num_evolutions=50,
        num_generations=25,
        num_islands=None,
        pop_size=10,
        seed=4444,
        algo=None,
    ),
)


def with_body_order(fn, body_order):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(body_order, *args, **kwargs)

    return wrapper


def get_leg_tof_bounds(destination, target):
    # in days
    time_map = {
        "short": [30, 365 * 1],
        "medium": [30 * 6, 365 * 3],
        "long": [365 * 5, 365 * 15],
    }

    bound_map = {
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
        destination, target = target, destination

    return [
        TimeDelta(x, format="jd").to_value("sec")
        for x in time_map[bound_map[(destination, target)]]
    ]


def generate_wishlist(body_orders, create_objs, p_kwargss, evolve_kwargss):
    wishlist = [
        merge(
            clone_deep(defaults),
            {
                "body_order": x[0],
                "create_obj": with_body_order(x[1], x[0]),
                "p_kwargs": x[2],
                "evolve_kwargs": x[3],
            },
        )
        for x in product(body_orders, create_objs, p_kwargss, evolve_kwargss)
    ]

    pprint(wishlist)

    def filter_wishlist(x):
        return True

    def map_wishlist(x):
        if "leg_tof" not in x["p_kwargs"]["bounds"]:
            num_legs = len(x["body_order"]) - 1
            x["p_kwargs"]["bounds"]["leg_tof"] = [
                get_leg_tof_bounds(x["body_order"][i - 1], x["body_order"][i])
                for i in range(1, num_legs + 1)
            ]
        return x

    def sort_wishlist(x):
        pass

    wishlist = [map_wishlist(x) for x in wishlist if filter_wishlist(x)]

    def generate_suffix(x):
        return f"_{x['create_obj'].__name__.replace('create_', '')}"

    for i, x in enumerate(wishlist):
        x["suffix"] = generate_suffix(x)

    return wishlist


def main(body_orders, create_objs, p_kwargss, evolve_kwargss):
    FOLDER = Path(__file__).parent / "runs"

    print("main:")
    print(f"body_orders: {body_orders}")
    print(f"create_objs: {create_objs}")
    print(f"p_kwargss: {p_kwargss}")
    print(f"evolve_kwargss: {evolve_kwargss}")

    if os.path.isfile((FOLDER / "wishlist.pkl").absolute()):
        with open((FOLDER / "wishlist.pkl").absolute(), "rb") as f:
            prev = pkl.load(f)
        print(f"Loaded wishlist with {len(prev)} entries, adding new entries to it!")
    else:
        prev = []

    wishlist = [
        *prev,
        *generate_wishlist(body_orders, create_objs, p_kwargss, evolve_kwargss),
    ]

    if len(wishlist) > 0:
        print(f"Generated wishlist with {len(wishlist)} entries")
        with open((FOLDER / "wishlist.pkl").absolute(), "wb") as f:
            pkl.dump(wishlist, f)


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

evolve_map = {
    "low": dict(
        num_evolutions=50,
        num_generations=25,
        pop_size=10,
    ),
    "high": dict(
        num_evolutions=600,
        num_generations=250,
        pop_size=300,
    ),
    "ultra": dict(
        num_evolutions=2000,
        num_generations=1000,
        pop_size=1000,
    ),
}


def parse_args():
    kwargs = dict(
        body_orders=body_orders[:1],
        create_objs=create_objs[:1],
        p_kwargss=p_kwargss,
        evolve_kwargss=evolve_kwargss,
    )

    args = sys.argv[1:]

    if len(args) > 0:
        create_arg = args[0].split(",")
        kwargs["create_objs"] = [create_map[x] for x in create_arg]

    if len(args) > 1:
        order_arg = args[1].split(",")
        kwargs["body_orders"] = [[body_map[x] for x in y] for y in order_arg]

    if len(args) > 2:
        evolve_arg = args[2].split(",")
        kwargs["evolve_kwargss"] = [evolve_map[x] for x in evolve_arg]

    print("Args: " + " x ".join([f"[{x}]" for x in sys.argv[1:]]))

    return kwargs


if __name__ == "__main__":
    kwargs = parse_args()
    main(**kwargs)


# %%
