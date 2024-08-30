# %%
import logging
import pygmo as pg
import multiprocessing as mp

from trajectory.lib.test_log_prob import Problem
from multiprocessing import Queue, Process
from logging.handlers import QueueHandler, QueueListener


def main():
    # logger = logging.getLogger()

    p = Problem()
    prob = pg.problem(p)
    algo = pg.algorithm(pg.nlopt("slsqp"))

    n = mp.cpu_count()
    # logger.info(f"n = {n}")

    # archi = pg.archipelago(n=n, algo=algo, prob=prob, pop_size=10, udi=pg.mp_island())
    # archi.evolve()
    # archi.wait_check()

    p.fitness([])
    pop = pg.population(prob, 10)
    pop = algo.evolve(pop)


if __name__ == "__main__":
    mp.freeze_support()

    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    #     handler.close()

    # log_queue = Queue(-1)
    # queue_handler = QueueHandler(log_queue)
    # handler = logging.StreamHandler()
    # handler = logging.FileHandler(logfile)
    # listener = QueueListener(log_queue, handler)
    # logger.addHandler(queue_handler)

    # handler.setFormatter(
    #     logging.Formatter(
    #         "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(funcName)s - %(message)s"
    #     )
    # )
    # listener.start()
    # logger.info("Starting main")

    main()

    # logger.info("Finished main")

    # listener.stop()
