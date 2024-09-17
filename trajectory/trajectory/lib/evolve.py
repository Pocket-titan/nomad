# %%
from pydash import merge
from tudatpy import constants
from tudatpy.astro.time_conversion import DateTime
from astropy.time import Time, TimeDelta
from pprint import pprint

import pygmo as pg
import pygmo_plugins_nonfree as ppnf
import polars as pl
import numpy as np
import multiprocessing as mp
import colorlog
import time
import os

from trajectory.lib.utils import is_notebook
from trajectory.lib.problem import Problem
from trajectory.lib.island import Island

logger = colorlog.getLogger(os.environ.get("LOG_NAME", None))

J2000 = Time("J2000", format="jyear_str")
MJD2000 = J2000.to_value("mjd")


def get_departure_arrival_time(x, number_of_legs):
    J2000 = Time("J2000", format="jyear_str")
    dep, arr = [
        (J2000 + x * TimeDelta(1, format="sec")).to_datetime()
        for x in [x[0], np.cumsum(x[: number_of_legs + 1])[-1]]
    ]
    return [x.strftime("%Y-%m-%d %H:%M") for x in [dep, arr]]


def convert_state_vector(x, number_of_legs):
    dep = (J2000 + x[0] * TimeDelta(1, format="sec")).to_value("mjd") - MJD2000
    legs = [TimeDelta(t, format="sec").to_value("jd") for t in x[1 : number_of_legs + 1]]
    print(
        np.array2string(np.array([dep, *legs]), precision=2, separator=",\n ", sign=" ")
    )


# start = Time(MJD2000 - 789.8055, format="mjd").to_value("datetime")
# flight_time = np.sum([158.33942, 449.38588, 54.720136, 1024.6563, 4552.7531])
# end = (Time(start) + TimeDelta(flight_time, format="jd")).to_value("datetime")
# print(start.strftime("%Y-%m-%d %H:%M"), ",", end.strftime("%Y-%m-%d %H:%M"))


# convert_state_vector([-6.83e07, 1.37e07, 3.88e07, 4.72e06, 8.86e07, 3.93e08], 5)

# convert_state_vector(
#     [-68266244.67, 13655472.87, 38826938.3, 4733704.39, 88469845.69, 393266282.94], 5
# )


algo_map = {
    "gaco": lambda kw: pg.gaco(**kw),
    "pso": lambda kw: pg.pso(**kw),
    "sga": lambda kw: pg.sga(**kw),
    "de": lambda kw: pg.de(**kw),
    "sade": lambda kw: pg.sade(**kw),
    "slsqp": lambda kw: pg.nlopt(**merge({"solver": "slsqp"}, kw)),
    "sa": lambda kw: pg.simulated_annealing(**kw),
}


# %%
def evolve(
    p: Problem,
    num_evolutions=50,
    num_generations=10,
    num_islands=None,
    pop_size=10,
    seed=4444,
    island=None,
    algo_name=None,
    algo_kwargs=None,
    **kwargs,
):
    # fmt: off
    if p.dim == 1:
        def gradient(self, dv):
            return pg.estimate_gradient(lambda x: self.fitness(x), dv, 1e-8)

        p.gradient = gradient.__get__(p)
    # fmt: on

    prob = pg.problem(p)

    if algo_name is None:
        if p.dim == 1:
            algo = pg.nlopt("slsqp")

            if hasattr(algo, "set_random_sr_seed"):
                algo.set_random_sr_seed(seed)

            algo = pg.mbh(algo, stop=5, perturb=0.01, seed=seed)

        if p.dim == 2:
            algo = pg.moead(gen=num_generations, seed=seed, **kwargs)
    else:
        if algo_name.startswith("mbh_"):
            inner_name = algo_name.split("_")[1]
            algo = pg.mbh(algo_map[inner_name](algo_kwargs["algo"]), **algo_kwargs["mbh"])
        else:
            algo = algo_map[algo_name](algo_kwargs or dict())

    if hasattr(algo, "set_random_sr_seed"):
        algo.set_random_sr_seed(seed)
    algo = pg.algorithm(algo)
    # algo.set_verbosity(1)

    num_islands = int(
        num_islands
        or os.environ.get("SLURM_CPUS_PER_TASK", max(1, min(32, mp.cpu_count())))
    )

    if island is None:
        island = Island(processes=num_islands)

    archi = pg.archipelago(
        n=num_islands,
        algo=algo,
        prob=prob,
        pop_size=pop_size,
        seed=seed,
        udi=island,
    )

    # log_interval = min(50, max(10, num_evolutions // 10))
    log_interval = 10

    results = dict(f=[], x=[])
    errs = []

    width = len(str(num_evolutions))
    t0 = time.perf_counter()

    for i in range(1, num_evolutions + 1):
        archi.evolve()
        archi.wait_check()

        best_f = np.inf
        best_x = None

        for island in archi:
            pop = island.get_population()

            results["f"].append(pop.get_f())
            results["x"].append(pop.get_x())
            errs.append(pop.problem.extract(Problem).errs)

            champion_x = pop.champion_x
            champion_f = pop.champion_f.item()
            if champion_f < best_f:
                best_x = champion_x
                best_f = champion_f

        if i == 1 or i % log_interval == 0 or i == num_evolutions:
            t = time.perf_counter() - t0
            logger.info(
                f"t: {t:>3.0f}s, evolution {i:{width}}/{num_evolutions}, best_f: \u001b[91m{best_f:5.0f}\u001b[22m\u001b[38;5;44m, best_x: {best_x}, departure, arrival: {get_departure_arrival_time(champion_x, p.number_of_legs)}"
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
        f"done running! champion_f: \u001b[91m{best_f:.0f}\u001b[22m\u001b[38;5;44m, champion_x: {best_x}"
    )
    logger.info(
        f"\tdeparture, arrival: {get_departure_arrival_time(best_x, p.number_of_legs)}"
    )

    logger.info(
        f"failed evaluations: {nerrs} out of {fevals}, or: {nerrs/fevals * 100:.0f}%"
    )
    logger.info(
        f"discarded: {oglen - newlen} out of {oglen}, or: {(oglen - newlen)/oglen * 100:.0f}%"
    )

    return df, errs, runtime, num_islands
