# %%
from tudatpy.astro.time_conversion import DateTime
from multiprocessing import freeze_support
from logging.handlers import QueueHandler
from argparse import ArgumentParser
from pathlib import Path

import pygmo_plugins_nonfree as ppnf
import cloudpickle as pkl
import pygmo as pg
import numpy as np
import colorlog
import logging
import os

from trajectory.lib.logger import setup_logger
from trajectory.lib.run import perform_run

# %%
f"{54.5 * 10 ** 6 * 10 ** 3:.2e}"
# %%
f"{108950e3 / 0.02:.2e}"
# %%
soi_s = 54.5 * 10**6 * 10**3
e_cass = 0.02
a_cass = 108950e3 / e_cass


print(a_cass / soi_s)
# %%
soi_n = 86.2 * 10**6 * 10**3
e_nomad = 0.02

a_nomad = 172320917.43119267 / e_nomad

# a_cass / soi_s * soi_n * e_nomad

a_nomad / soi_n
