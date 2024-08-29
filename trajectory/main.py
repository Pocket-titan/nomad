# %%
from typing import Callable, Any
from multiprocessing import freeze_support
from pathlib import Path
import numpy as np
import logging
import time

import pygmo as pg
import pygmo_plugins_nonfree as ppnf
from tudatpy.astro.time_conversion import DateTime
from trajectory.lib.run import perform_run


logger = logging.getLogger(__name__)


body_order = ["Earth", "Mars", "Jupiter", "Neptune"]


def create_obj(*pars):
    from tudatpy.numerical_simulation.environment_setup import create_simplified_system_of_bodies
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


node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]

p_kwargs = dict(
    bounds=dict(
        departure=[DateTime(2028, 4, 6).epoch(), DateTime(2033, 12, 31).epoch()],
        leg_tof=np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]],
        node_parameters={},
        leg_parameters={},
    ),
    fixed=dict(
        departure=[],
        leg_tof=[],
        node_parameters={},
        leg_parameters={},
    ),
    dim=1,
)

evolve_kwargs = dict(
    num_evolutions=50,
    num_generations=25,
    pop_size=10,
    seed=4444,
)
evolve_kwargs["algo"] = pg.mbh(ppnf.snopt7(), stop=5, perturb=0.01, seed=evolve_kwargs["seed"])

suffix = "_test"


def main():
    FOLDER = Path(__file__).parent / "runs"

    logging.basicConfig(
        filename=(FOLDER / "main.log").absolute(),
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting main")

    freeze_support()

    # args

    perform_run(
        body_order=body_order,
        create_obj=create_obj,
        p_kwargs=p_kwargs,
        evolve_kwargs=evolve_kwargs,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()

# %%
