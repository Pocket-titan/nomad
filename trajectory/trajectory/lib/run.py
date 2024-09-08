from typing import Callable
from pathlib import Path
from astropy.time import Time, TimeDelta
from pydash import omit

import cloudpickle as pkl
import polars as pl
import pygmo as pg
import logging
import json
import time
import os

from trajectory.lib.island import Island
from trajectory.lib.problem import Problem
from trajectory.lib.evolve import evolve
from trajectory.lib.utils import truncate


logger = logging.getLogger(os.environ.get("LOG_NAME", None))

J2000 = Time("J2000", format="jyear_str")


def typename(x):
    _type = type(x)
    return f"{_type.__module__}.{_type.__name__}"


class Run:
    def __init__(
        self,
        p: Problem,
        df: pl.LazyFrame,
        num_errs: int,
        evolve_kwargs: dict,
        body_order: list[str],
        runtime: float,
    ) -> None:
        self.p = p
        self.df = df
        self.num_errs = num_errs
        self.body_order = body_order
        self.evolve_kwargs = omit(evolve_kwargs, ["island"])

        self.runtime = float(f"{runtime:.2f}")

        algo_name = ""
        algo = self.evolve_kwargs.pop("algo", None)
        if algo is None:
            algo_name = "default"
        elif isinstance(algo, str):
            algo_name = algo
        else:
            algo = algo()
            try:
                algo_name = repr(pg.algorithm(algo))
            except Exception:
                if hasattr(algo, "inner_algorithm"):
                    algo_name = f"{typename(algo)}_{algo.inner_algorithm.get_name()}"
                elif hasattr(algo, "get_solver_name"):
                    algo_name = f"{typename(algo)}_{algo.get_solver_name()}"
                else:
                    algo_name = typename(algo)

        self.algo = [x.replace("\t", "    ") for x in algo_name.split("\n")]

    def champion(self) -> dict:
        try:
            champion = (
                self.df.filter(pl.col("dv") == pl.min("dv"))
                .head(1)
                .collect()
                .to_dicts()[0]
            )

            departure_date = J2000 + TimeDelta(champion["x"][0], format="sec")
            arrival_date = J2000 + TimeDelta(
                sum(champion["x"][: self.p.number_of_legs + 1]), format="sec"
            )

            return truncate(
                {
                    "departure_date": departure_date.to_value("iso", subfmt="date"),
                    "arrival_date": arrival_date.to_value("iso", subfmt="date"),
                    **champion,
                }
            )
        except Exception:
            return {}

    def write(self, folder: Path, run_name: str):
        os.makedirs((folder / run_name).absolute(), exist_ok=True)

        self.df.collect().write_parquet((folder / run_name / "df.parquet").absolute())

        with open((folder / run_name / "info.json").absolute(), "w") as f:
            json.dump(
                {
                    "runtime": self.runtime,
                    "body_order": self.body_order,
                    "num_errs": self.num_errs,
                    "champion": self.champion(),
                    "evolve_kwargs": {**self.evolve_kwargs, "algo": self.algo},
                },
                f,
                indent=4,
            )

        with open((folder / run_name / "problem.pkl").absolute(), "wb") as f:
            pkl.dump(self.p, f)

    @classmethod
    def read(cls, folder: Path, run_name: str):
        df = pl.scan_parquet((folder / run_name / "df.parquet").absolute())

        with open((folder / run_name / "info.json").absolute(), "r") as f:
            info = json.load(f)

        with open((folder / run_name / "problem.pkl").absolute(), "rb") as f:
            p = pkl.load(f)

        return cls(
            p=p,
            df=df,
            num_errs=info["num_errs"],
            runtime=info["runtime"],
            body_order=info["body_order"],
            evolve_kwargs=info["evolve_kwargs"],
        )


def make_run_name(run: Run, suffix="") -> str:
    t = time.localtime()
    month, day, hour, minute = t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min

    order = "".join([x[0].upper() for x in run.body_order])

    num = (run.champion() or dict(dv=-1))["dv"]
    dv = f"{num/10:.1e}".replace(".", "").replace("+", "")

    return f"{month:02d}_{day:02d}_{hour:02d}h{minute:02d}m_{order}_{dv}_{run.evolve_kwargs['num_islands']}{suffix}"


def perform_run(
    body_order: list[str],
    create_obj: Callable,
    p_kwargs: dict,
    evolve_kwargs: dict,
    suffix="",
    queue=None,
    initializer=None,
) -> Run:
    p = Problem(create_obj, **p_kwargs)

    island = None
    if queue is not None and initializer is not None:
        processes = os.environ.get("SLURM_CPUS_PER_TASK", None)
        island = Island(processes, queue, initializer)

    print("evolving...")
    df, errs, runtime, num_islands = evolve(p, **evolve_kwargs, island=island)

    print("performing run...")
    run = Run(
        p=p,
        df=df.lazy(),
        num_errs=sum(errs),
        body_order=body_order,
        evolve_kwargs={**evolve_kwargs, "num_islands": num_islands},
        runtime=runtime,
    )

    print("making run...")
    name = make_run_name(run, suffix=suffix)

    logger.info(f"Finished run '{name}' after running for {runtime:.2f} seconds")

    FOLDER = Path(__file__).parents[2] / "runs"
    run.write(FOLDER, name)

    logger.info(f"Successfully saved run '{name}'")
