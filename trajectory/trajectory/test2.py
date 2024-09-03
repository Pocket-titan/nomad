# %%
import multiprocessing as mp
import logging
import logging.handlers
from logging.handlers import QueueHandler, QueueListener
import time


def setup_logger():
    pass


def worker_init(queue):
    qh = logging.handlers.QueueHandler(queue)
    logger = logging.getLogger("worker")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def worker(i):
    logger = logging.getLogger("worker")
    logger.debug(f"Hello from worker: {i}")
    time.sleep(5 - i)
    logger.debug(f"Goodbye from worker: {i}")


def listener_process(queue):
    pass


def main():
    mp.set_start_method("spawn", force=True)
    queue = mp.Manager().Queue(-1)

    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    queue_handler = QueueHandler(queue)
    listener = QueueListener(queue, handler)
    logger.addHandler(queue_handler)
    listener.start()

    # listener = mp.Process(target=listener_process, args=(queue,))
    # listener.start()

    logger.info("Main started")

    n = 5
    pool = mp.Pool(n, initializer=worker_init, initargs=(queue,))

    for i in range(n):
        pool.apply(worker, args=(i,))

    pool.close()
    pool.terminate()

    queue.put_nowait(None)
    # listener.join()
    listener.stop()

    # process = mp.Process(target=worker, args=(queue,))
    # process.start()
    # print(queue.get())
    # process.join()


if __name__ == "__main__":
    main()
