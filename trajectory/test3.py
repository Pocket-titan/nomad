# %%
import multiprocessing as mp
import pygmo as pg
import logging
import logging.handlers
import time
import wat

from logging.handlers import QueueHandler, QueueListener
from trajectory.lib.test_log_problem import Problem
from trajectory.lib.island import Island

logger = logging.getLogger()


def initializer(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    # root.setLevel(logging.DEBUG)


def fn(queue, initializer, i, sleep_time):
    initializer(queue)

    # def wrapper():
    # logger = logging.getLogger("problem")
    logging.info(f"hello from {i}!")
    time.sleep(sleep_time)

    # return wrapper


def mp_stuff(queue):
    mp_ctx = mp.get_context("fork")
    pool = mp_ctx.Pool(3)

    job_list = [1, 2, 3, 4, 5, 6]

    # getters = []
    # for i, sleep_time in enumerate(job_list):
    #     name = str(i)
    #     with Island._pool_lock:
    #         res = pool.apply_async(fn, args=(queue, initializer, name, sleep_time))
    #     getters.append(res)

    # while len(getters):
    #     getters.pop().get()
    # # optionally, close and join pool here (generally a good idea anyway)
    # queue.put_nowait(None)

    for i, sleep_time in enumerate(job_list):
        name = str(i)

        with Island._pool_lock:
            res = pool.apply_async(fn, args=(queue, initializer, name, sleep_time))

    pool.close()
    pool.join()
    queue.put_nowait(None)


def main(queue):
    logger.info("Start")

    # mp_stuff(queue)

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
        udi=Island(initializer=lambda: initializer(queue)),
    )

    for i in range(3):
        archi.evolve()
        archi.wait_check()
        logger.info(f"Generation {i}")

    print(archi.get_champions_f())
    print(prob.get_fevals())

    logger.info("End")


def setup_logger():
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


def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def listener_configurer():
    root = logging.getLogger()
    log_file = "test.log"
    h = logging.FileHandler(log_file)
    # h = logging.StreamHandler()
    f = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    h.setFormatter(f)
    root.addHandler(h)


if __name__ == "__main__":
    # queue, listener = setup_logger()
    queue = mp.Manager().Queue(-1)
    listener = mp.Process(target=listener_process, args=(queue, listener_configurer))
    listener.start()

    main(queue)

    # queue.put_nowait(None)
    listener.join()

    # logging.shutdown()

# %%
