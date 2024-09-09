# %%
from difflib import SequenceMatcher
from threading import Thread
import multiprocessing as mp
import logging.handlers
import logging
import traceback
import sys
import re


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


class LogFilter(logging.Filter):
    def __init__(self, name: str = "", threshold=0.8) -> None:
        super().__init__(name)
        self.threshold = threshold
        self._reset()
        self.times_flushed = 0

    def _preprocess(self, msg: str):
        out = re.sub(r"\(repeated ([0-9]*) times\)", "", msg)
        out = re.sub(r"\d+", "", out)
        return out.strip()

    def _reset(self):
        self.last_message = None
        self.count = 0
        self._name = "root"
        self._pathname = ""
        self._lvl = logging.INFO
        self._func = ""

    def _save(self, record: logging.LogRecord):
        self.last_message = record.msg
        self._name = record.name
        self._pathname = record.pathname
        self._lvl = record.levelno
        self._func = record.funcName

    def filter(self, record):
        if self.last_message is None:
            self._save(record)
            return True

        similarity = SequenceMatcher(
            None,
            self._preprocess(self.last_message),
            self._preprocess(record.msg),
        ).ratio()

        if similarity >= self.threshold:
            self.count += 1
            self._lvl = max(self._lvl, record.levelno)
            self._name = record.name
            self._pathname = record.pathname
            self._func = record.funcName
            record.msg = f"{self.last_message} (repeated {self.count + 1} times)"
            return False
        else:
            self.last_message = record.msg
            record.msg = f"{self.last_message} (repeated {self.count + 1} times)"
            self._reset()
            return True

    def getname(self):
        return self._name

    def getpathname(self):
        return self._pathname

    def getlevel(self):
        return self._lvl

    def getfunc(self):
        return self._func

    def _flush(self, record):
        if self.count > 0:
            record.msg = f"{self.last_message} (repeated {self.count + 1} times)"
            self._reset()
            return True

        return False


class StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.filters = []

    def flush(self):
        for filter in self.filters:
            if isinstance(filter, LogFilter):
                name = filter.getname()
                level = int(filter.getlevel())
                pathname = filter.getpathname()
                func = filter.getfunc()

                record = logging.LogRecord(
                    name, level, pathname, 0, "", (), None, func=func
                )

                if filter._flush(record):
                    self.emit(record)

        super().flush()


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(
        self,
        filename,
        mode="a",
        maxBytes=0,
        backupCount=0,
        encoding=None,
        delay=False,
        errors=None,
    ):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay, errors)
        self.filters = []

    def flush(self):
        for filter in self.filters:
            if isinstance(filter, LogFilter):
                pass
                name = filter.getname()
                level = int(filter.getlevel())
                pathname = filter.getpathname()
                func = filter.getfunc()

                record = logging.LogRecord(
                    name, level, pathname, 0, "", (), None, func=func
                )

                if filter._flush(record):
                    self.emit(record)

        super().flush()

    def doRollover(self):
        self.flush()
        super().doRollover()


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

    handlers = [
        StreamHandler(sys.stdout),
        RotatingFileHandler(filename, maxBytes=5_000_000, backupCount=5),
    ]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(LogFilter())
        logger.addHandler(handler)

    logger.setLevel(level)

    queue = mp.Manager().Queue(-1)
    reader = LogReader(queue, stop_sign=stop_sign)

    return reader, queue, logger
