# %%
from pathlib import Path
from tudatpy.astro.time_conversion import DateTime
from astropy.time import Time, TimeDelta
from tudatpy.util import result2array
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import polars as pl
import numpy as np

from trajectory.lib.run import Run
from trajectory.lib.utils import get_dates

sns.set_theme(style="ticks", palette="deep")

FOLDER = Path(__file__).parents[1]


def read_runs(folder):
    run_names = [x.name for x in folder.iterdir() if x.is_dir()]
    runs = [Run.read(folder, name) for name in run_names]
    return runs


def filter_runs(runs):
    ignore_list = [
        "slsqp",
        "mbh_slsqp",
        "sa",
        "sga",
        "mbh_sga",
    ]
    return [x for x in runs if x.evolve_kwargs["algo_name"] not in ignore_list]


def plot_best(folder, title=None):
    if title is None:
        title = Path(folder).name

    runs = read_runs(folder)

    fig, axes = plt.subplots(ncols=2, figsize=(8, 5), layout="constrained")
    fig.suptitle(title)

    for run in runs:
        best_fs = (
            run.df.filter(pl.col("gen") > 10)
            .group_by("gen")
            .agg(pl.all().sort_by("dv").first())
            .sort("gen")
            .collect()
        )

        (line,) = axes[0].plot(
            best_fs["gen"], best_fs["dv"], label=run.evolve_kwargs["algo_name"]
        )

        overgens = run.df.filter(
            (pl.col("gen") > 10) & (pl.col("dv") <= 15_000)
        ).collect()

        mins = overgens.group_by("gen").agg(pl.min("dv")).sort("gen")["dv"]
        maxs = overgens.group_by("gen").agg(pl.max("dv")).sort("gen")["dv"]

        axes[1].fill_between(
            overgens["gen"].unique(maintain_order=True),
            y1=maxs,
            y2=mins,
            color=line.get_color(),
            alpha=0.25,
        )

    axes[0].legend()


# %%
def panel_plot(folder, title=None, c="gen"):
    if title is None:
        title = Path(folder).name

    runs = read_runs(folder)

    order = [
        # ["gaco", "mbh_gaco"],
        # ["pso", "mbh_pso"],
        # ["sade", "mbh_sade"],
        ["sade"],
    ]

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    subfigs = fig.subfigures(nrows=1, ncols=2 + 1, width_ratios=[1, 1, 0.2])

    nrows = len(order)
    ncols = len(order[0])

    gs0 = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=subfigs[0])
    gs1 = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=subfigs[1])

    axes0 = gs0.subplots(sharex=True, sharey=True)
    axes1 = gs1.subplots(sharex=True, sharey=True)

    if not isinstance(axes0, np.ndarray):
        axes0 = np.array([[axes0]])
        axes1 = np.array([[axes1]])

    fig.suptitle(title)
    subfigs[0].suptitle("times")
    subfigs[1].suptitle("dv")

    minc, maxc = np.inf, 0
    for run in runs:
        df = run.df.sort(pl.col("dv"))
        minc = min(minc, df.select(pl.col(c)).min().collect().item() or minc)
        maxc = max(maxc, df.select(pl.col(c)).max().collect().item() or maxc)
    norm = plt.Normalize(vmin=minc / 1000, vmax=maxc / 1000)

    for run in runs:
        algo = run.evolve_kwargs["algo_name"]

        try:
            i, j = [(i, x.index(algo)) for i, x in enumerate(order) if algo in x][0]
        except (IndexError, ValueError):
            continue

        axes0[i, j].set_title(algo)
        axes1[i, j].set_title(algo)

        df = run.df.sort(pl.col("dv")).collect()
        times = get_dates(df["x"], run.p.number_of_legs)[..., [0, -1]]

        axes0[i, j].scatter(
            times[:, 0],
            times[:, 1],
            c=df[c] / 1000,
            label=algo,
            cmap="viridis",
            norm=norm,
        )

        axes1[i, j].scatter(
            times[:, 0],
            df["dv"] / 1000,
            c=df[c] / 1000,
            label=algo,
            cmap="viridis",
            norm=norm,
        )

        for ax in [axes0[i, j], axes1[i, j]]:
            sbpl = ax.get_subplotspec()

            if sbpl.is_last_row():
                ax.set_xlabel("Departure date")
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)

    subfigs[2].colorbar(
        cm.ScalarMappable(norm=norm, cmap="viridis"),
        label=r"$\Delta$V [km/s]",
        cax=subfigs[2].subplots(1, 1),
    )


# # %%
# panel_plot(FOLDER / "gen50")
# plot_best(FOLDER / "gen50")


# # %%
# panel_plot(FOLDER / "pop100")
# plot_best(FOLDER / "pop100")

# # %%
# panel_plot(FOLDER / "ev200pop100", c="dv")
# plot_best(FOLDER / "ev200pop100")

# # %%
# panel_plot(FOLDER / "baseline", c="dv")
# plot_best(FOLDER / "baseline")

# # %%
# panel_plot(FOLDER / "wide", c="dv")
# plot_best(FOLDER / "wide")

# # %%
# panel_plot(FOLDER / "narrow", c="dv")
# plot_best(FOLDER / "narrow")

# # %%
# panel_plot(FOLDER / "neptune_wide_unpowered", c="dv")
# plot_best(FOLDER / "neptune_wide_unpowered")

# %%
panel_plot(FOLDER / "neptune_unpowered", c="dv")
plot_best(FOLDER / "neptune_unpowered")


# %%
def create_unpowered(
    *pars,
    body_order=None,
    departure_semi_major_axis=np.inf,
    departure_eccentricity=0.0,
    arrival_semi_major_axis=3.8913e9,
    arrival_eccentricity=0.999486,
    minimum_pericenters=None,
):
    from tudatpy.numerical_simulation.environment_setup import (
        create_simplified_system_of_bodies,
    )
    from tudatpy.trajectory_design.transfer_trajectory import (
        create_transfer_trajectory,
        mga_settings_unpowered_unperturbed_legs,
    )

    central_body = "Sun"

    bodies = create_simplified_system_of_bodies()

    leg_settings, node_settings = mga_settings_unpowered_unperturbed_legs(
        body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
        **(
            dict(minimum_pericenters=minimum_pericenters)
            if minimum_pericenters is not None
            else {}
        ),
    )

    obj = create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )

    return obj, leg_settings, node_settings


# %%
# runs = read_runs(FOLDER / "neptune_wide_unpowered")
runs = read_runs(FOLDER / "neptune_unpowered")
run = runs[0]

champion_x = run.df.sort("dv").head(10).collect().select(pl.col("x"))[0].item().to_numpy()
run.p.evaluate(champion_x)
obj = run.p.obj(*champion_x)
obj.delta_v

# %%
ranges = np.linspace(1e6, 1e10, 1000)
dvs = []

e = 0.999999999

for r in ranges:
    obj, _, _ = create_unpowered(
        body_order=run.body_order,
        departure_semi_major_axis=np.inf,
        departure_eccentricity=0.0,
        # arrival_semi_major_axis=1e6,
        arrival_semi_major_axis=r,
        arrival_eccentricity=e,
    )

    # res = [*champion_x[:-1], champion_x[-1] * 0.9]

    obj.evaluate(*run.p.convert_parameters(champion_x))
    dvs.append(obj.delta_v)

# print(obj.delta_v)
# print(obj.delta_v_per_leg)
# print(obj.delta_v_per_node)

plt.plot(dvs)

a = ranges[np.argmin(dvs)]
soi = 86.2e3 * 10**6

print(f"e = {e}")
print(f"a = {a:.3e} m")
print(f"soi = {soi:.3e} m")
print(f"dv = {min(dvs):.0f} m/s")

# %%
from tudatpy.interface import spice

kernels = Path(__file__).parents[1] / "spice"

spice.load_standard_kernels()
spice.load_kernel(str((kernels / "de440s.bsp").absolute()))
spice.load_kernel(str((kernels / "nep097.bsp").absolute()))
spice.load_kernel(str((kernels / "jup344-s2003_j24.bsp").absolute()))
spice.load_kernel(str((kernels / "mar097.bsp").absolute()))


state_history = obj.states_along_trajectory(500)
states = result2array(state_history)

state_keys = np.fromiter(state_history.keys(), dtype=float)


bodies = ["Neptune", "Earth", "Jupiter"]

for body in bodies:
    body_state_history = {
        epoch: spice.get_body_cartesian_position_at_epoch(
            body, "Sun", "J2000", "NONE", epoch
        )
        for epoch in state_keys
    }

    body_states = result2array(body_state_history)

    plt.plot(body_states[:, 1], body_states[:, 2], label=body)


plt.plot(states[:, 1], states[:, 2])
plt.scatter([0], [0], label="Sun")
plt.legend()

# %%


# %%
def plot_orbit(obj, run):
    state_history = obj.states_along_trajectory(500)
    states = result2array(state_history)

    champion = run.df.sort("dv").first().collect()
    champion_x = champion["x"].item().to_numpy()

    state_keys = np.fromiter(state_history.keys(), dtype=float)
    node_times = np.cumsum(champion_x[..., : run.p.number_of_legs + 1], axis=-1)
    node_time_indices = (np.abs(state_keys - node_times[:, None])).argmin(axis=1)
    fly_bys = [state_history[state_keys[x]] for x in node_time_indices]

    plt.plot(states[:, 1], states[:, 2])

    for i, fly_by in enumerate(fly_bys):
        plt.scatter(fly_by[0], fly_by[1], label=run.body_order[i], zorder=3)

    plt.scatter([0], [0], label="Sun")
    plt.legend()


plot_orbit(obj, run)
obj.delta_v_per_node

# %%
