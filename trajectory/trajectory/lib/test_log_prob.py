# %%
import pygmo as pg
import logging


# logger = logging.getLogger()


class Problem(pg.rosenbrock):
    def __init__(self, queue=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.init_logger(queue)

    def init_logger(self, queue):
        h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

    def fitness(self, x):
        # logger = logging.getLogger("problem")
        # logger.info(f"fitness: {x}")
        print(f"fitness: {x}")
        return super().fitness(x)

    def gradient(self, x):
        return super().gradient(x)

    def get_bounds(self):
        return super().get_bounds()

    def get_nix(self):
        return super().get_nix()

    def get_nec(self):
        return super().get_nec()

    def get_nic(self):
        return super().get_nic()
