# %%
from tudatpy.astro.time_conversion import DateTime
from multiprocessing import freeze_support
from pathlib import Path

from multiprocessing import Queue, Process
from logging.handlers import QueueHandler, QueueListener

import pygmo_plugins_nonfree as ppnf
import cloudpickle as pkl
import pygmo as pg
import numpy as np
import logging

from trajectory.lib.run import perform_run


# logger = logging.getLogger(__name__)


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
    ppnf.snopt7(), stop=5, perturb=0.01, seed=evolve_kwargs["seed"]
)

suffix = "_test"


# Define a queue


# Define a log listener in the main process
def log_listener(queue):
    handler = logging.StreamHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    listener = QueueListener(queue, handler)
    listener.start()
    try:
        listener.join()
    except KeyboardInterrupt:
        pass
    listener.stop()


def listener_process(queue, configurer):
    pass


def listener_configurer():
    pass


def main():
    FOLDER = Path(__file__).parent / "runs"

    logfile = (FOLDER / "main.log").absolute()
    # if not logfile.exists():
    #     logfile.touch()

    # logger = logging.getLogger()

    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    #     handler.close()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    log_queue = Queue(-1)
    queue_handler = QueueHandler(log_queue)
    handler = logging.FileHandler(logfile)
    listener = QueueListener(log_queue, handler)
    logger.addHandler(queue_handler)

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(funcName)s - %(message)s"
        )
    )
    listener.start()

    # logging.FileHandler((FOLDER / "main.log").absolute())

    # logfile = (FOLDER / "main.log").absolute()
    # if not logfile.exists():
    #     logfile.touch()
    # logging.basicConfig(
    #     filename=logfile,
    #     level=logging.DEBUG,
    #     format="%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(funcName)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     force=True,
    # )

    freeze_support()

    try:
        with open((FOLDER / "wishlist.pkl").absolute(), "rb") as f:
            wishlist: list[dict] = pkl.load(f)
    except Exception as e:
        logger.error("Failed to load wishlist", exc_info=e)
        return

    logger.info(f"Loaded wishlist with {len(wishlist)} entries")

    # perform_run(
    #     body_order=body_order,
    #     create_obj=create_obj,
    #     p_kwargs=p_kwargs,
    #     evolve_kwargs=evolve_kwargs,
    #     suffix=suffix,
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
                log_queue=log_queue,
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

    listener.stop()


if __name__ == "__main__":
    main()
