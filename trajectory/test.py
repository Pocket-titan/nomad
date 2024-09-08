from tudatpy.astro.time_conversion import DateTime
from multiprocessing import freeze_support
from logging.handlers import QueueHandler
from pathlib import Path

import pygmo_plugins_nonfree as ppnf
import cloudpickle as pkl
import pygmo as pg
import numpy as np
import logging
import os

from trajectory.lib.logger import setup_logger
from trajectory.lib.run import perform_run


if __name__ == "__main__":
    print("hello")
