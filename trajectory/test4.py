# %%
import os
import sys
import time
import traceback
import multiprocessing as mp
import threading
import logging
import sys

DEFAULT_LEVEL = logging.DEBUG

formatter = logging.Formatter(
    "%(levelname)s: %(asctime)s - %(name)s - %(process)s - %(message)s"
)


class SubProcessLogHandler(logging.Handler):
    """handler used by subprocesses

    It simply puts items on a Queue for the main process to log.

    """

    def __init__(self, queue):
        logging.Handler.__init__(self)
        self.queue = queue

    def emit(self, record):
        self.queue.put(record)


class LogQueueReader(threading.Thread):
    """thread to write subprocesses log records to main process log

    This thread reads the records written by subprocesses and writes them to
    the handlers defined in the main process's handlers.

    """

    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        """read from the queue and write to the log handlers

        The logging documentation says logging is thread safe, so there
        shouldn't be contention between normal logging (from the main
        process) and this thread.

        Note that we're using the name of the original logger.

        """
        # Thanks Mike for the error checking code.
        while True:
            try:
                record = self.queue.get()

                if record is True:
                    break
                # get the logger for this record
                logger = logging.getLogger(record.name)
                logger.callHandlers(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)


class LoggingProcess(mp.Process):
    def __init__(self, queue):
        mp.Process.__init__(self)
        self.queue = queue

    def _setupLogger(self):
        # create the logger to use.
        logger = logging.getLogger("test.subprocess")
        # The only handler desired is the SubProcessLogHandler.  If any others
        # exist, remove them. In this case, on Unix and Linux the StreamHandler
        # will be inherited.

        for handler in logger.handlers:
            # just a check for my sanity
            assert not isinstance(handler, SubProcessLogHandler)
            logger.removeHandler(handler)
        # add the handler
        handler = SubProcessLogHandler(self.queue)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # On Windows, the level will not be inherited.  Also, we could just
        # set the level to log everything here and filter it in the main
        # process handlers.  For now, just set it from the global default.
        logger.setLevel(DEFAULT_LEVEL)
        self.logger = logger

    def run(self):
        self._setupLogger()
        logger = self.logger
        # and here goes the logging
        # p = mp.current_process()
        # logger.info("hello from process %s with pid %s" % (p.name, p.pid))
        logger.info("HM?")


def configurer(queue):
    logger = logging.getLogger("test.subprocess")

    add = True
    for handler in logger.handlers:
        if isinstance(handler, SubProcessLogHandler):
            add = False
        else:
            logger.removeHandler(handler)

    if add:
        handler = SubProcessLogHandler(queue)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # On Windows, the level will not be inherited.  Also, we could just
    # set the level to log everything here and filter it in the main
    # process handlers.  For now, just set it from the global default.
    logger.setLevel(DEFAULT_LEVEL)
    return logger


def worker(queue, configurer, i, sleep_time):
    logger = configurer(queue)

    p = mp.current_process()
    logger.debug(f"{i}: WHAT, {sleep_time}")
    time.sleep(sleep_time)
    return i
    # return i
    # logger.info("hello from process %s with pid %s" % (p.name, p.pid))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # queue used by the subprocess loggers
    queue = mp.Manager().Queue(-1)
    # Just a normal logger
    logger = logging.getLogger("test")

    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(DEFAULT_LEVEL)
    logger.info("hello from the main process")
    # This thread will read from the subprocesses and write to the main log's
    # handlers.
    log_queue_reader = LogQueueReader(queue)
    log_queue_reader.start()
    # create the processes.

    with mp.get_context().Pool(2) as pool:
        sleep_times = [0.5, 1, 1.5, 2]

        # pool.map(worker, [(queue, configurer, i, sleep_time) for i, sleep_time in enumerate(sleep_times)])

        getters = []

        # for i, sleep_time in enumerate(sleep_times):
        #     # p = LoggingProcess(queue)
        #     # p.start()
        #     getters.append(
        #         pool.apply_async(worker, args=(queue, configurer, i, sleep_time))
        #     )

        getters = [
            pool.apply_async(worker, args=(queue, configurer, i, sleep_time))
            for i, sleep_time in enumerate(sleep_times)
        ]

        # while len(getters):
        #     print(len(getters))
        #     x = getters.pop().wait()

        print([res.wait() for res in getters])
        print([res.successful() for res in getters])
        print([res.get() for res in getters])

        # The way I read the mp warning about Queue, joining a
        # process before it has finished feeding the Queue can cause a deadlock.
        # Also, Queue.empty() is not realiable, so just make sure all processes
        # are finished.
        # active_children joins subprocesses when they're finished.
        # while mp.active_children():
        #     time.sleep(0.1)

        pool.close()
        pool.join()
        queue.put_nowait(True)

# %%
