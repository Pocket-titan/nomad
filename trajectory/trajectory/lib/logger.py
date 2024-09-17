# %%
from difflib import SequenceMatcher
from threading import Thread
import multiprocessing as mp
import logging.handlers
import logging
import colorlog
import colorlog
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

                logger = colorlog.getLogger(record.name)
                logger.handle(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except Exception:
                traceback.print_exc(file=sys.stderr)


class LogFilter(logging.Filter):
    def __init__(self, name: str = "", threshold=0.95) -> None:
        super().__init__(name)
        self.threshold = threshold
        self._reset()

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
        if record.levelno not in [logging.WARNING, logging.ERROR]:
            if self.last_message is not None:
                self._reset()
            return True

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
    logger = colorlog.getLogger(name)

    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    fmt = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    datefmt = "%m-%d %H:%M"

    handlers = [
        colorlog.StreamHandler(sys.stdout),
        RotatingFileHandler(filename, maxBytes=5_000_000, backupCount=5),
    ]

    handlers[0].setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(time_log_color)s%(asctime)s%(time_log_color)s - %(log_color)s%(levelname)s%(time_log_color)s - %(func_log_color)s%(funcName)s%(time_log_color)s - %(message_log_color)s%(message)s",
            datefmt="%m-%d %H:%M",
            log_colors={
                "DEBUG": "blue",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={
                "message": {
                    "DEBUG": "light_blue",
                    "INFO": "light_cyan",
                    "WARNING": "light_yellow",
                    "ERROR": "light_red",
                    "CRITICAL": "light_purple",
                },
                "func": {
                    "DEBUG": "light_purple",
                    "INFO": "light_purple",
                    "WARNING": "light_purple",
                    "ERROR": "light_purple",
                    "CRITICAL": "light_purple",
                },
                "time": {
                    "DEBUG": "light_black",
                    "INFO": "light_black",
                    "WARNING": "light_black",
                    "ERROR": "light_black",
                    "CRITICAL": "light_black",
                },
            },
        )
    )
    handlers[1].setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
            datefmt="%m-%d %H:%M",
        )
    )

    for handler in handlers:
        handler.addFilter(LogFilter())
        logger.addHandler(handler)

    logger.setLevel(level)

    queue = mp.Manager().Queue(-1)
    reader = LogReader(queue, stop_sign=stop_sign)

    return reader, queue, logger
