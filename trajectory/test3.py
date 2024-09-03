# %%
import multiprocessing as mp
import pygmo as pg
import logging
import logging.handlers
import wat

from logging.handlers import QueueHandler, QueueListener
from trajectory.lib.test_log_problem import Problem

from threading import Lock as _Lock


class Island(object):
    _pool_lock = _Lock()
    _pool_size = None
    _pool = None

    def __init__(self):
        self._init()

    def _init(self):
        self.init_pool()

    @staticmethod
    def init_pool(processes=None):
        with Island._pool_lock:
            Island._init_pool_impl(processes)

    @staticmethod
    def _init_pool_impl(processes):
        from ._mp_utils import _make_pool

        if Island._pool is None:
            Island._pool, Island._pool_size = _make_pool(processes)

    @staticmethod
    def get_pool_size():
        with Island._pool_lock:
            Island._init_pool_impl(None)
            return Island._pool_size

    @staticmethod
    def resize_pool(processes):
        from ._mp_utils import _make_pool

        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError("The 'processes' argument must be strictly positive")

        with Island._pool_lock:
            Island._init_pool_impl(processes)
            if processes == Island._pool_size:
                return

            new_pool, new_size = _make_pool(processes)

            Island._pool.close()
            Island._pool.join()

            Island._pool = new_pool
            Island._pool_size = new_size

    @staticmethod
    def shutdown_pool():
        with Island._pool_lock:
            if Island._pool is None:
                return

            Island._pool.close()
            Island._pool.join()
            Island._pool = None
            Island._pool_size = None

    def get_name(self):
        return "Multiprocessing island (custom)"

    def get_extra_info(self):
        retval = "\tUsing a process pool: {}\n".format("yes")
        retval += "\tNumber of processes in the pool: {}".format(
            mp_island.get_pool_size()
        )
        return retval

    def __copy__(self):
        return Island()

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        return

    def __setstate__(self, state):
        self._init()

    def run_evolve(self, algo, pop):
        pass


class mp_island(pg.mp_island):
    def __init__(self, use_pool=True):
        super().__init__(use_pool)

    @staticmethod
    def init_pool(self):
        print("no")


def main():
    logger = logging.getLogger()
    logger.info("Start")

    p = Problem(2)
    prob = pg.problem(p)

    seed = 42
    algo = pg.de(gen=1)
    algo = pg.algorithm(algo)

    n = 2
    archi = pg.archipelago(
        n=n,
        algo=algo,
        prob=prob,
        pop_size=5,
        seed=seed,
        udi=mp_island(),
    )

    for i in range(3):
        archi.evolve()
        archi.wait_check()
        logger.info(f"Generation {i}")

    logger.info("End")
    print(archi.get_champions_f())
    print(prob.get_fevals())


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler()
    queue = mp.Manager().Queue(-1)
    queue_handler = QueueHandler(queue)
    listener = QueueListener(queue, handler)
    logger.addHandler(queue_handler)

    return queue, listener


if __name__ == "__main__":
    queue, listener = setup_logger()
    listener.start()
    main()
    listener.stop()
    logging.shutdown()

# %%
