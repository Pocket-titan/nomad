# %%
from threading import Thread
import multiprocessing as mp
import logging.handlers
import logging
import traceback
import sys


class LogReader(Thread):
    def __init__(self, queue: mp.Queue, *args, stop_sign=None, **kwargs) -> None:
        super().__init__(*args, daemon=True, **kwargs)
        self.stop_sign = stop_sign
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
    stop_sign=True,
) -> tuple[LogReader, mp.Queue, logging.Logger]:
    logger = logging.getLogger(name)

    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(filename)
    # handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)

    queue = mp.Manager().Queue(-1)
    reader = LogReader(queue, stop_sign=stop_sign)

    return reader, queue, logger
