# %%
import pygmo as pg
import pygmo_plugins_nonfree as ppnf
import polars as pl
import numpy as np
import multiprocessing as mp
import logging
import time
import os

from tudatpy import constants
from trajectory.lib.utils import is_notebook
from trajectory.lib.problem import Problem
from trajectory.lib.island import Island

logger = logging.getLogger(os.environ.get("LOG_NAME", None))


def evolve(
    p: Problem,
    num_evolutions=50,
    num_generations=10,
    num_islands=None,
    pop_size=10,
    seed=4444,
    algo=None,
    island=None,
    **kwargs,
):
    # fmt: off
    if p.dim == 1:
        def gradient(self, dv):
            return pg.estimate_gradient(lambda x: self.fitness(x), dv, 1e-8)

        p.gradient = gradient.__get__(p)
    # fmt: on

    prob = pg.problem(p)

    if algo is None:
        if p.dim == 1:
            # algo = pg.nlopt(solver="slsqp")
            algo = pg.nlopt("slsqp")
            # algo = pg.pso(gen=num_generations, seed=seed)

            if hasattr(algo, "set_random_sr_seed"):
                algo.set_random_sr_seed(seed)

            algo = pg.mbh(algo, stop=5, perturb=0.01, seed=seed)

        if p.dim == 2:
            algo = pg.moead(gen=num_generations, seed=seed, **kwargs)
    else:
        algo = algo()

    if hasattr(algo, "set_random_sr_seed"):
        algo.set_random_sr_seed(seed)
    algo = pg.algorithm(algo)

    num_islands = int(
        os.environ.get("SLURM_CPUS_PER_TASK", num_islands)
        or max(1, min(32, mp.cpu_count()))
    )
    print([*os.environ.items()])
    print(f"num_islands: {num_islands}")

    if island is None:
        island = Island()

    print("archi")
    archi = pg.archipelago(
        n=num_islands,
        algo=algo,
        prob=prob,
        pop_size=pop_size,
        seed=seed,
        udi=island,
    )
    print("after archi")

    results = dict(f=[], x=[])
    errs = []

    width = len(str(num_evolutions))
    t0 = time.perf_counter()

    for i in range(1, num_evolutions + 1):
        archi.evolve()
        archi.wait_check()

        best = np.inf

        for island in archi:
            pop = island.get_population()

            results["f"].append(pop.get_f())
            results["x"].append(pop.get_x())
            errs.append(pop.problem.extract(Problem).errs)

            champion_f = pop.champion_f.item()
            if champion_f < best:
                best = champion_f

        if i == 1 or i % 50 == 0 or i == num_evolutions:
            t = time.perf_counter() - t0
            logger.info(
                f"t: {t:>3.0f}s, evolution {i:{width}}/{num_evolutions}, best: {best:5.0f}"
            )

    runtime = time.perf_counter() - t0
    evolutions = [[i] * pop_size * num_islands for i in range(1, num_evolutions + 1)]
    fs = np.concatenate(results["f"])
    xs = np.concatenate(results["x"])

    tof = np.cumsum(xs[:, : p.obj(xs[0]).number_of_legs + 1], axis=-1)
    tof = (tof[:, -1] - tof[:, 0]) / constants.JULIAN_YEAR

    df = pl.DataFrame(
        {
            "dv": fs[:, 0],
            "tof": tof,
            "x": xs,
            "gen": np.concatenate(evolutions),
        }
    )
    oglen = len(df)
    df = df.filter((pl.col("dv") != p.death_value))
    newlen = len(df)

    nerrs = pop.problem.extract(Problem).errs
    fevals = np.sum([x.get_population().problem.get_fevals() for x in archi])

    logger.info(
        f"failed evaluations: {nerrs} out of {fevals}, or: {nerrs/fevals * 100:.0f}%"
    )
    logger.info(
        f"discarded: {oglen - newlen} out of {oglen}, or: {(oglen - newlen)/oglen * 100:.0f}%"
    )

    return df, errs, runtime, num_islands
