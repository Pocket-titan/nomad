# %%
from typing import Optional, Union, Any, Callable
from astropy.time import Time, TimeDelta
from pprint import pprint
from itertools import groupby

import cloudpickle as pkl
import multiprocessing as mp
import pygmo_plugins_nonfree as ppnf
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pygmo as pg
import time
import re
import wat

from multiprocessing import freeze_support
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

from trajectory.lib.utils import flatten, dictify, once, is_notebook
from trajectory.lib.problem import Problem
from trajectory.lib.optimize import evolve


np.set_printoptions(precision=2, suppress=True)
sns.set_theme(style="ticks", palette="deep")
pal = sns.color_palette("deep")


@once
def print_parameters(leg_settings, node_settings):
    print_parameter_definitions(leg_settings, node_settings)


def create_obj(*pars):
    from tudatpy.numerical_simulation.environment_setup import create_simplified_system_of_bodies
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        mga_settings_unpowered_unperturbed_legs,
    )

    central_body = "Sun"

    body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

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

    # print("Parameters:")
    # print_parameters(leg_settings, node_settings)

    obj = create_transfer_trajectory(
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


p = Problem(
    create_obj,
    bounds=bounds,
    fixed=fixed,
    dim=1,
)

# %%
x = pkl.loads(pkl.dumps(p))
wat(x)
# %%
from pathlib import Path


if __name__ == "__main__":
    freeze_support()

    FOLDER = Path(__file__).parent / "runs"

    df, errs = evolve(
        p,
        num_evolutions=50,
        num_generations=50,
        pop_size=60,
        seed=4444,
    )


# %%
import os
import sys


sys.version_info
# %%
# FOLDER = os.path.join(os.path.dirname(__file__), "runs")


FOLDER = Path(__file__).parent / "runs"

# %%


class Run:
    def __init__(
        self,
        p: Problem,
        df: pl.LazyFrame,
        errs: list[float],
        evolve_kwargs: dict,
        body_order: list[str],
        runtime: float,
        algo: Any,
    ) -> None:
        self.p = p
        self.df = df
        self.errs = errs
        self.runtime = runtime
        self.body_order = body_order
        self.evolve_kwargs = evolve_kwargs

        if isinstance(algo, str):
            self.algo = algo
        elif hasattr(algo, "get_solver_name"):
            self.algo = f"{repr(algo)}_{algo.get_solver_name()}"
        else:
            self.algo = repr(algo)

    def champion():
        pass

    def write(self, folder: Path, name: str):
        os.mkdir((folder / name).asolute(), exist_ok=True)
        self.df.collect().write_parquet((folder / name / "df.parquet").absolute())

    @classmethod
    def read(cls, folder: Path, name: str):
        df = pl.scan_parquet(name)


run = Run(
    p, df.lazy(), errs, {}, ["Earth", "Mars", "Jupiter", "Neptune"], create_obj, 0.0, pg.moead()
)
# len(pkl.dumps(run)) / 1024, df.estimated_size("kb")

# # %%
# # %%

# %%
df = pl.LazyFrame(
    {
        "x": [0, 1, 2, 3],
        "y": [4, 5, 6, 7],
        "n": ["a", "b", "c", "d"],
        "dv": [0, 1, 2, 3],
    }
)


# df.filter(pl.col("x") > 1).collect()
# get row where dv is min
q = df.filter(pl.col("dv") == pl.min("dv")).head(1)
q.collect().to_dicts()[0]

# %%
from pathlib import Path

(Path(__file__)).parents[0]
# copilot: can i write the above line more concisely?
# answer:
# Path(__file__).parent.parent.parent
