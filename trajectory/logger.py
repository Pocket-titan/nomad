# %%
from logging.handlers import QueueHandler
from threading import Thread
import multiprocessing as mp
import logging.handlers
import logging
import traceback
import time
import sys
import os

from trajectory.lib.test_log_problem import Problem
from trajectory.lib.island import Island
import pygmo as pg


class LogReader(Thread):
    def __init__(self, queue: mp.Queue, *args, stop_sign=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stop_sign = stop_sign
        self.daemon = True
        self.queue = queue

    def run(self) -> None:
        while True:
            try:
                record = self.queue.get()

                if record == self.stop_sign:
                    break

                logger = logging.getLogger(record.name)
                logger.handle(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except Exception:
                traceback.print_exc(file=sys.stderr)


def setup_logger(
    name: str = None,
    level=logging.DEBUG,
    filename: str = "main.log",
) -> tuple[LogReader, mp.Queue, logging.Logger]:
    logger = logging.getLogger(name)

    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # handler = logging.FileHandler(filename)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)

    queue = mp.Manager().Queue(-1)
    reader = LogReader(queue, stop_sign=True)

    return reader, queue, logger


def worker(i, sleep_time):
    logger = logging.getLogger("test.subprocess")

    p = mp.current_process()
    logger.debug(f"{i}: WHAT {p.pid}, {sleep_time}")
    time.sleep(sleep_time)
    return i


def configurer(queue):
    logger = logging.getLogger("test.subprocess")

    add = True
    for handler in logger.handlers:
        if isinstance(handler, QueueHandler):
            add = False
        else:
            logger.removeHandler(handler)

    if add:
        handler = QueueHandler(queue)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)
    # return logger


def main(queue):
    logger = logging.getLogger("test")
    logger.info("Starting main")

    p = Problem(dim=2)
    prob = pg.problem(p)
    algo = pg.algorithm(pg.compass_search())

    archi = pg.archipelago(
        n=3,
        algo=algo,
        prob=prob,
        pop_size=10,
        udi=Island(queue, configurer),
    )

    for i in range(5):
        logger.info(f"Generation {i}")
        archi.evolve()
        archi.wait_check()

    logger.info("Ending main")
    logger.info(f"Champion: {archi.get_champions_f()}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    reader, queue, logger = setup_logger("test")
    reader.start()

    # start
    main(queue)
    # end

    queue.put_nowait(reader.stop_sign)

# %%
