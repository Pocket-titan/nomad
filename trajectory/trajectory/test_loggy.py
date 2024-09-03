# %%
import logging
import pygmo as pg
import multiprocessing as mp

from trajectory.lib.test_log_problem import Problem
from multiprocessing import Queue, Process
from logging.handlers import QueueHandler, QueueListener
import cloudpickle as pkl
import wat

import threading
import time


def setup_logger():
    logger = logging.getLogger("multiprocessing_logger")
    logger.setLevel(logging.INFO)
    #  QueueHandler to send log records to a logging queue
    queue_handler = logging.handlers.QueueHandler(mp.Queue(-1))
    logger.addHandler(queue_handler)
    return logger


def worker():
    logger = setup_logger()
    logger.info("Hello")


def main():
    logger = logging.getLogger()

    p = Problem()
    prob = pg.problem(p)
    algo = pg.algorithm(pg.nlopt("slsqp"))

    n = mp.cpu_count()
    logger.info(f"n = {n}")

    archi = pg.archipelago(
        n=n,
        algo=algo,
        prob=prob,
        pop_size=10,
        udi=pg.mp_island(),
    )
    archi.evolve()
    archi.wait_check()

    pop = pg.population(prob, 10)
    pop = algo.evolve(pop)


def listener_process(log_queue):
    root = logging.getLogger()
    handler = logging.StreamHandler()
    # Add a handler to output logs to the console
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    while True:
        try:
            # Get log record from the queue
            record = log_queue.get()
            # Check for the termination signal (None)
            if record is None:
                break
            # Get the logger specified by the record and process the log message
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys, traceback

            print("Error in logger process", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    log_queue = Queue(-1)
    # queue_handler = QueueHandler(log_queue)
    # handler = logging.StreamHandler()
    # # handler = logging.FileHandler(logfile)
    # listener = QueueListener(log_queue, handler)
    # logger.addHandler(queue_handler)

    listener = Process(target=listener_process, args=(log_queue,))

    # handler.setFormatter(
    #     logging.Formatter(
    #         "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(funcName)s - %(message)s"
    #     )
    # )
    listener.start()

    processes = []
    for i in range(5):
        p = Process(target=worker)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info("Finished main")

    log_queue.put_nowait(None)
    listener.join()

# %%
