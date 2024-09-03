# %%
import pygmo as pg
import numpy as np
import logging


logger = logging.getLogger("problem")


class Problem:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        logger.info(f"Fitness {x}")

        retval = np.zeros((1,))
        for i in range(len(x) - 1):
            retval[0] += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        return retval

    def get_bounds(self):
        return (np.full((self.dim,), -5.0), np.full((self.dim,), 10.0))
