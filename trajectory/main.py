# %%
from tudatpy.astro.time_conversion import DateTime
from multiprocessing import freeze_support
from logging.handlers import QueueHandler
from pathlib import Path

import pygmo_plugins_nonfree as ppnf
import cloudpickle as pkl
import pygmo as pg
import numpy as np
import logging
import os

from trajectory.lib.logger import setup_logger
from trajectory.lib.run import perform_run


body_order = ["Earth", "Mars", "Jupiter", "Neptune"]


def create_obj(*pars):
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
evolve_kwargs["algo"] = lambda: pg.mbh(
    pg.nlopt("slsqp"), stop=5, perturb=0.01, seed=evolve_kwargs["seed"]
)

suffix = "_test"


# Main
# For now, use the root logger - if you change this probably have to use an .env file to propagate it to the other modules
LOG_NAME = "root"
LOG_LEVEL = logging.DEBUG
SUBP_LOG_NAME = f"{LOG_NAME}.subp"


def initializer(queue):
    logger = logging.getLogger(SUBP_LOG_NAME)

    add = True
    for handler in logger.handlers:
        if isinstance(handler, QueueHandler):
            add = False
        else:
            logger.removeHandler(handler)

    if add:
        handler = QueueHandler(queue)
        logger.addHandler(handler)

    logger.setLevel(LOG_LEVEL)


def main():
    freeze_support()

    os.environ["LOG_NAME"] = LOG_NAME
    os.environ["SUBP_LOG_NAME"] = SUBP_LOG_NAME

    FOLDER = Path(__file__).parent / "runs"
    FOLDER.mkdir(exist_ok=True)

    logfile = (FOLDER / "main.log").absolute()
    reader, queue, logger = setup_logger(LOG_NAME, level=LOG_LEVEL, filename=logfile)
    reader.start()

    try:
        with open((FOLDER / "wishlist.pkl").absolute(), "rb") as f:
            wishlist: list[dict] = pkl.load(f)
    except Exception as e:
        logger.error("Failed to load wishlist", exc_info=e)
        return

    logger.info(f"Loaded wishlist with {len(wishlist)} entries")

    kwargs = dict(
        queue=queue,
        initializer=initializer,
    )

    # perform_run(
    #     body_order=body_order,
    #     create_obj=create_obj,
    #     p_kwargs=p_kwargs,
    #     evolve_kwargs=evolve_kwargs,
    #     suffix=suffix,
    #     **kwargs,
    # )

    completed = []
    for i, w in enumerate(wishlist):
        logger.info(
            f"Running wishlist entry {i}. {''.join([x[0].upper() for x in w['body_order']])}, {w['create_obj'].__name__}, {', '.join([str(k) + '=' + str(v) for k, v in w['evolve_kwargs'].items()])}"
        )

        try:
            perform_run(
                body_order=w["body_order"],
                create_obj=w["create_obj"],
                p_kwargs=w["p_kwargs"],
                evolve_kwargs=w["evolve_kwargs"],
                suffix=w["suffix"],
                **kwargs,
            )
            completed.append(i)
        except Exception as e:
            logger.error(f"Failed to run wishlist entry {i}", exc_info=e)

    logger.info("Finished running wishlist")

    wishlist_after = [i for i in range(len(wishlist)) if i not in completed]
    with open((FOLDER / "wishlist_after.pkl").absolute(), "wb") as f:
        pkl.dump(wishlist_after, f)

    if len(wishlist_after) == 0:
        logger.info("All wishlist entries completed successfully")
    else:
        logger.warning(
            f"Failed to complete some wishlist entries: {len(wishlist_after)} out of {len(wishlist)}, or {len(wishlist_after)/len(wishlist)*100:.2f}%"
        )

    queue.put_nowait(reader.stop_sign)


if __name__ == "__main__":
    main()

# %%
