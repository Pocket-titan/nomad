# %%
from pathlib import Path
from tudatpy.astro.time_conversion import DateTime
from tudatpy.interface import spice
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from tudatpy import constants
from astropy.time import Time, TimeDelta
from tudatpy.util import result2array
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import polars as pl
import numpy as np

from trajectory.lib.run import Run
from trajectory.lib.utils import get_dates
from wishlist import create_unpowered, create_1dsm

sns.set_theme(
    style="ticks",
    palette="deep",
    rc={
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.color": "lightgray",
        "grid.alpha": 0.5,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
    },
)
deep = sns.color_palette("deep")
set2 = sns.color_palette("Set2")

pal1 = sns.cubehelix_palette(start=0.4, gamma=0.6, hue=1.2, rot=-0.75, light=0.6, dark=0.4, as_cmap=True)
pal2 = sns.cubehelix_palette(start=0.6, gamma=0.7, hue=1.2, rot=-0.75, light=0.5, dark=0.27, as_cmap=True)
pal3 = sns.cubehelix_palette(start=0.8, gamma=0.75, hue=1.2, rot=-0.75, light=0.4, dark=0.18, as_cmap=True)

aurora = colors.LinearSegmentedColormap.from_list(
    "aurora", [*zip([0, 0.33, 0.66, 1], [pal1(0), pal1(pal1.N), pal2(pal2.N), pal3(pal3.N)])]
)

kernels = Path(__file__).parents[1] / "spice"

spice.load_standard_kernels()
spice.load_kernel(str((kernels / "de440s.bsp").absolute()))
spice.load_kernel(str((kernels / "nep097.bsp").absolute()))
spice.load_kernel(str((kernels / "jup344-s2003_j24.bsp").absolute()))
spice.load_kernel(str((kernels / "mar097.bsp").absolute()))

FOLDER = Path(__file__).parents[1]


def read_runs(folder):
    run_names = [x.name for x in folder.iterdir() if x.is_dir()]
    runs = [Run.read(folder, name) for name in run_names]
    return runs


runs = read_runs(FOLDER / "neptune_1dsm")

MU_SUN = spice.get_body_gravitational_parameter("Sun")
MU_EARTH = spice.get_body_gravitational_parameter("Earth")
MU_NEPTUNE = spice.get_body_gravitational_parameter("Neptune")
R_EARTH = spice.get_average_radius("Earth")

a_arr = 1.9203e9
e_arr = 0.999999

r_dep = 185e3 + R_EARTH
a_dep = r_dep
e_dep = 0.0

body_map = {
    "E": "Earth",
    "N": "Neptune",
    "J": "Jupiter",
    "M": "Mars",
    "V": "Venus",
}


# %%
def approx_unique(df):
    return (
        df.filter(pl.col("traj_dv") <= 10_000)
        .with_columns(
            [
                pl.col("departure").dt.round(every="6d").alias("round_departure"),
                pl.col("arrival").dt.round(every="6d").alias("round_arrival"),
                pl.col("tof").round_sig_figs(2).alias("round_tof"),
                pl.col("traj_dv").round_sig_figs(3).alias("round_traj_dv"),
            ]
        )
        .group_by(
            [
                "body_order",
                "round_tof",
                "round_traj_dv",
                "round_departure",
                "round_arrival",
            ]
        )
        .agg(pl.all().sort_by("traj_dv").head(1))
        .explode(
            pl.exclude(
                [
                    "body_order",
                    "round_tof",
                    "round_traj_dv",
                    "round_departure",
                    "round_arrival",
                ]
            )
        )
        .collect()
    )


df1 = pl.scan_parquet(FOLDER / "trajectories_unpowered_2.parquet")
df2 = pl.scan_parquet(FOLDER / "trajectories_1dsm_2.parquet")

df1 = approx_unique(df1)
df2 = approx_unique(df2)


# %%
# plt.scatter(df1["departure"], df1["arrival"], c=df1["traj_dv"])
# plt.scatter(df2["departure"], df2["arrival"], c=df2["traj_dv"])

c1 = ["#1d1640", "#3d276f", "#001d83", "#39977f", "#b4dac3"]
c2 = ["#283a6c", "#015869", "#7088c6", "#00b2a4", "#b5b9dc"]
c3 = ["#002D42", "#002D42", "#6DD2DA", "#92CC6F"]

marker_map = {
    "EJN": "o",
    "EMJN": "s",
    "EN": "^",
}

min_tof = min(df1["tof"].min(), df1["tof"].min())
max_tof = max(df1["tof"].max(), df1["tof"].max())
norm = plt.Normalize(min_tof, max_tof)

pts = []

for body_order in df2["body_order"].unique():
    df = df2.filter(pl.col("body_order") == body_order).with_columns(
        [pl.lit(marker_map[body_order]).alias("marker")]
    )
    pts.append(df)

for body_order in df1["body_order"].unique():
    df = df1.filter(pl.col("body_order") == body_order).with_columns(
        [pl.lit(marker_map[body_order]).alias("marker")]
    )
    pts.append(df)

df = (
    pl.concat(pts)
    .group_by(["round_tof", "marker"], maintain_order=True)
    .agg(pl.all())
    .sort("round_tof", descending=True)
    .explode(pl.all().exclude(["round_tof", "marker"]))
)

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained", dpi=300)
cmap = aurora.reversed()


for i, (name, data) in enumerate(df.group_by(["round_tof", "marker"], maintain_order=True)):
    x = data.sort("tof", descending=True)

    ax.scatter(
        x["departure"],
        x["traj_dv"] / 1000,
        s=25,
        alpha=1,
        c=x["tof"],
        cmap=cmap,
        norm=norm,
        edgecolors=colors.colorConverter.to_rgba("white", alpha=0.25),
        linewidths=0.075,
        marker=name[1],
        zorder=i + 2,
    )

plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="time of flight [yr]", ax=ax)
plt.xlabel("Departure year")
plt.ylabel(r"$\Delta$v [km/s]")
# plt.title("Trajectories")


handles = []
for name, marker in marker_map.items():
    handles.append(
        mlines.Line2D(
            [],
            [],
            linewidth=0,
            color=cmap(cmap.N // 4),
            marker=marker,
            markersize=5,
            label=name,
        )
    )

plt.legend(
    handles=handles,
    loc="lower left",
    bbox_to_anchor=(0.21, 0.05),
    fontsize=10,
    handletextpad=0,
    labelspacing=0.35,
)

# plt.savefig("trajectories_dv_departure.png", bbox_inches="tight", dpi=300)
# plt.savefig("trajectories_dv_departure.pdf", bbox_inches="tight", dpi=300)
# plt.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x * 1e-3:.1f}"))

# %%
res = (
    df.with_columns(
        [
            pl.col("traj_dv").round_sig_figs(2).alias("r_traj_dv"),
            pl.col("departure").dt.round(every="7d").alias("r_departure"),
            pl.col("arrival").dt.round(every="7d").alias("r_arrival"),
        ]
    )
    .group_by(
        [
            "r_traj_dv",
            "r_departure",
            "r_arrival",
            "body_order",
        ]
    )
    .agg(pl.all().sort_by("traj_dv").head(1))
    .explode(pl.exclude(["r_traj_dv", "r_departure", "r_arrival", "body_order"]))
).drop(["r_traj_dv", "r_departure", "r_arrival"])

res = (
    res.group_by(["round_traj_dv", "marker"], maintain_order=True)
    .agg(pl.all())
    .sort("round_traj_dv", descending=True)
    .explode(pl.all().exclude(["round_traj_dv", "marker"]))
)

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained", dpi=300)
cmap = aurora.reversed()
norm = plt.Normalize(res["traj_dv"].min() / 1000, res["traj_dv"].max() / 1000)


for i, (name, data) in enumerate(res.group_by(["round_traj_dv", "marker"], maintain_order=True)):
    x = data.sort("traj_dv", descending=True)

    ax.scatter(
        x["departure"],
        x["arrival"],
        s=25,
        alpha=1,
        c=x["traj_dv"] / 1000,
        cmap=cmap,
        norm=norm,
        edgecolors=colors.colorConverter.to_rgba("white", alpha=0.25),
        linewidths=0.075,
        marker=name[1],
        zorder=i + 2,
    )


plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label=r"$\Delta$v [km/s]", ax=ax)
plt.xlabel("Departure year")
plt.ylabel("Arrival year")

handles = []
for name, marker in marker_map.items():
    handles.append(
        mlines.Line2D(
            [],
            [],
            linewidth=0,
            color=cmap(cmap.N // 4),
            marker=marker,
            markersize=5,
            label=name,
        )
    )

plt.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=10,
    handletextpad=0,
    labelspacing=0.35,
)

# plt.savefig("trajectories_departure_arrival.png", bbox_inches="tight", dpi=300)
# plt.savefig("trajectories_departure_arrival.pdf", bbox_inches="tight", dpi=300)

# %%
res2 = (
    df.with_columns(
        [
            pl.col("traj_dv").round_sig_figs(2).alias("r_traj_dv"),
            pl.col("tof").round_sig_figs(2).alias("r_tof"),
            (pl.col("traj_dv") / df["traj_dv"].min() + pl.col("tof") / df["tof"].min()).alias("ratio"),
            pl.col("departure").dt.round(every="14d").alias("r_departure"),
            pl.col("arrival").dt.round(every="14d").alias("r_arrival"),
        ]
    )
    .group_by(
        [
            "r_traj_dv",
            "r_tof",
            "body_order",
            "kind",
            "r_departure",
            "r_arrival",
        ]
    )
    .agg(pl.all().sort_by("ratio").head(1))
    .explode(pl.exclude(["r_traj_dv", "r_tof", "body_order", "kind", "r_departure", "r_arrival"]))
).drop(["r_traj_dv", "r_tof", "r_departure", "r_arrival"])


res2 = (
    res2.group_by(["round_traj_dv", "marker"], maintain_order=True)
    .agg(pl.all())
    .sort("round_traj_dv", descending=True)
    .explode(pl.all().exclude(["round_traj_dv", "marker"]))
)

# Cluster
cluster_data = []

for name, data in res2.group_by(["marker", "kind"]):
    marker = name[0]
    kind = name[1]

    pts = data.select(["tof", "traj_dv"]).to_numpy()
    # scaler = MinMaxScaler((0, 1))
    # X = scaler.fit_transform(pts)

    db = DBSCAN(eps=0.75, min_samples=2).fit(pts)

    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise_ = list(db.labels_).count(-1)

    cluster_mask = np.zeros_like(db.labels_, dtype=bool)
    cluster_mask[db.core_sample_indices_] = True

    for label in set(db.labels_):
        if label == -1:
            singles = pts[(db.labels_ == label) & ~cluster_mask]
            x, y = singles.T
            size = 1
            cluster_data.extend(zip(x, y, [marker] * len(x), [kind] * len(x), [size] * len(x)))
        else:
            clusters = pts[(db.labels_ == label) & cluster_mask]
            x, y = clusters.mean(axis=0)
            size = len(clusters)
            cluster_data.append((x, y, marker, kind, size))


cluster_df = (
    pl.DataFrame(cluster_data, schema=["tof", "traj_dv", "marker", "kind", "size"], orient="row")
    .with_columns(
        [
            pl.col("traj_dv").round_sig_figs(2).alias("r_traj_dv"),
            pl.col("tof").round_sig_figs(2).alias("r_tof"),
        ]
    )
    .group_by(["r_traj_dv", "r_tof", "marker", "kind"], maintain_order=True)
    .agg(pl.all())
    .sort("r_traj_dv", descending=True)
    .explode(pl.exclude(["r_traj_dv", "r_tof", "marker", "kind"]))
    .drop(["r_traj_dv", "r_tof"])
)

print(f"Went from {res2.shape[0]} pts to {len(cluster_data)} clusters. Max size: {max(cluster_df['size'])}")

# End cluster


fig, ax = plt.subplots(figsize=(7, 4), layout="constrained", dpi=300)
cmap = aurora.reversed()


for i, (name, data) in enumerate(cluster_df.group_by(["marker", "kind"], maintain_order=True)):
    x = data.sort(["traj_dv", "tof"], descending=True)

    ax.scatter(
        x["tof"],
        x["traj_dv"] / 1000,
        s=9 + ((1 * (x["size"] - 1)) ** (0.35)) * 2,
        alpha=0.65,
        edgecolors=colors.colorConverter.to_rgba("white", alpha=0.25),
        # edgecolors="none",
        linewidths=0.1,
        marker=name[0],
        c="#39977f" if name[1] == "unpowered" else "#b53dff",
        zorder=i + 2,
    )

plt.xlabel("Time of flight [yr]")
plt.ylabel(r"$\Delta$v [km/s]")

handles = []
for name, color in [("unpowered", "#39977f"), ("1dsm", "#b53dff")]:
    handles.append(
        mlines.Line2D(
            [],
            [],
            linewidth=0,
            color=color,
            marker="o",
            markersize=5,
            label=name,
        )
    )

plt.legend(
    handles=handles,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    fontsize=10,
    handletextpad=0,
    labelspacing=0.35,
)

# plt.savefig("trajectories_dv_tof.png", bbox_inches="tight", dpi=300)
# plt.savefig("trajectories_dv_tof.pdf", bbox_inches="tight", dpi=300)

# %%
# - dv vs departure (c=tof, m=body_order)
# - arrival vs departure (c=dv, m=body_order)
# - dv vs tof (c=kind, m=body_order)

res3 = (
    df.sort("departure")
    .with_columns(pl.col("departure").alias("_departure"))
    .group_by_dynamic("departure", every="1y")
    .agg(pl.all().sort_by("traj_dv").head(1))
    .explode(pl.exclude("departure"))
    .drop(["round_tof", "round_traj_dv", "round_arrival"])
    .with_columns(
        pl.col("departure").alias("bin_departure"),
        # pl.col("round_departure").alias("departure"),
    )
)


fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6.5, 3 * 2), layout="constrained", dpi=300)
cmap = aurora.reversed()
tof_norm = plt.Normalize(res3["tof"].min(), res3["tof"].max())
dv_norm = plt.Normalize(res3["traj_dv"].min() / 1000, res3["traj_dv"].max() / 1000)

for name, data in res3.group_by("marker"):
    marker = name[0]

    axes[0].scatter(
        data["departure"],
        data["arrival"],
        s=60 if marker == "o" else 85,
        alpha=1,
        c=data["traj_dv"] / 1000,
        cmap=cmap,
        norm=dv_norm,
        edgecolors=colors.colorConverter.to_rgba(((0.35,) * 3), alpha=0.25),
        linewidths=0.5,
        marker=marker,
        zorder=5,
    )

    axes[1].scatter(
        data["departure"],
        data["traj_dv"] / 1000,
        s=60 if marker == "o" else 85,
        alpha=1,
        c=data["tof"],
        cmap=cmap,
        norm=tof_norm,
        edgecolors=colors.colorConverter.to_rgba(((0.35,) * 3), alpha=0.25),
        linewidths=0.5,
        marker=marker,
        zorder=5,
    )

axes[0].plot(res3["departure"], res3["arrival"], color=cmap(cmap.N // 4), alpha=0.5, lw=1.5)

axes[1].plot(res3["departure"], res3["traj_dv"] / 1000, color=cmap(cmap.N // 4), alpha=0.5, lw=1.5)

axes[0].set_ylabel("Arrival year")
axes[1].set_ylabel(r"$\Delta$v [km/s]")
axes[1].set_xlabel("Departure year")


# Annotations with arrows
# Point 1: Annotating bottom-left points (e.g., first point)
axes[1].annotate(
    "",
    xy=(0.043, 0.05),
    xytext=(0.065, 0.6),
    arrowprops=dict(
        facecolor=((0.25,) * 3), shrink=0.05, width=1.5, headwidth=7, headlength=7, edgecolor="none", alpha=1
    ),
    xycoords="axes fraction",
    textcoords="axes fraction",
    fontsize=9,
    ha="center",
)

# Point 2: Annotating bottom-left (e.g., second point)
axes[1].annotate(
    "",
    xy=(0.105, 0.1),
    xytext=(0.065, 0.6),
    arrowprops=dict(
        facecolor=((0.25,) * 3), shrink=0.05, width=1.5, headwidth=7, headlength=7, edgecolor="none", alpha=1
    ),
    xycoords="axes fraction",
    textcoords="axes fraction",
    fontsize=9,
    ha="center",
)

axes[1].text(
    0.065,
    0.62,
    "Prime",
    fontsize=9,
    ha="center",
    va="center",
    color=(0.25,) * 3,
    alpha=1,
    transform=axes[1].transAxes,
)

# Points around 2038 - 2040
# Example point in that range
axes[1].annotate(
    "",
    xy=(0.59, 0.075),
    xytext=(0.6, 0.5),
    arrowprops=dict(
        facecolor=((0.25,) * 3), shrink=0.05, width=1.5, headwidth=7, headlength=7, edgecolor="none", alpha=1
    ),
    xycoords="axes fraction",
    textcoords="axes fraction",
    fontsize=9,
    ha="center",
)

axes[1].annotate(
    "",
    xy=(0.65, 0.075),
    xytext=(0.6, 0.5),
    arrowprops=dict(
        facecolor=((0.25,) * 3), shrink=0.05, width=1.5, headwidth=7, headlength=7, edgecolor="none", alpha=1
    ),
    xycoords="axes fraction",
    textcoords="axes fraction",
    fontsize=9,
    ha="center",
)

axes[1].text(
    0.6,
    0.51,
    "Backup",
    fontsize=9,
    ha="center",
    va="center",
    color=(0.25,) * 3,
    alpha=1,
    transform=axes[1].transAxes,
)

plt.colorbar(cm.ScalarMappable(norm=dv_norm, cmap=cmap), label=r"$\Delta$v [km/s]", ax=axes[0])
plt.colorbar(cm.ScalarMappable(norm=tof_norm, cmap=cmap), label="time of flight [yr]", ax=axes[1])

handles = []
for name, marker in [("EJN", "o"), ("EN", "^")]:
    handles.append(
        mlines.Line2D(
            [],
            [],
            linewidth=0,
            color=cmap(cmap.N // 4),
            marker=marker,
            markersize=5,
            label=name,
        )
    )

axes[1].legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    fontsize=10,
    handletextpad=0,
    labelspacing=0.35,
)


# plt.savefig("trajectories_lowest_yearly.png", bbox_inches="tight", dpi=300)
# plt.savefig("trajectories_lowest_yearly.pdf", bbox_inches="tight", dpi=300)

# %%
res4 = (
    df.sort("departure")
    .with_columns(pl.col("departure").alias("min_departure"))
    .group_by_dynamic("departure", every="3mo", closed="left")
    .agg(pl.all().sort_by("traj_dv").head(1))
    .explode(pl.exclude("departure"))
).sort("traj_dv")


champion = res4.head(1)

obj, _, _ = create_1dsm(
    body_order=[body_map[x] for x in [*champion["body_order"].to_numpy()[0]]],
    departure_semi_major_axis=a_dep,
    departure_eccentricity=e_dep,
    arrival_semi_major_axis=a_arr,
    arrival_eccentricity=e_arr,
    minimum_pericenters={
        "Earth": 6371e3 * 1.5,
        "Venus": 6051.8e3 * 1.5,
        "Jupiter": 69_911e3 * 1.5,
        "Saturn": 58_232e3 * 1.5,
        "Mars": 3389.5e3 * 1.5,
    },
    dsm_leg_index=champion["dsm_leg_index"].to_numpy()[0],
)

run_names = [run.name for run in runs]
run = runs[run_names.index(champion["run_name"].to_numpy()[0])]

x = champion["x"].to_numpy().item()
node_times, leg_parameters, node_parameters = run.p.convert_parameters(x)
obj.evaluate(node_times, leg_parameters, node_parameters)

state_history = obj.states_along_trajectory(500)
states = result2array(state_history)

state_keys = np.fromiter(state_history.keys(), dtype=float)
node_time_indices = (np.abs(state_keys - node_times[:, None])).argmin(axis=1)
flybys = [state_history[state_keys[x]] for x in node_time_indices]

fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300, layout="constrained")


bodies = ["Earth", "Jupiter", "Neptune"]

colors = {
    "Sun": "#ffd700",
    "Neptune": "#00bfff",
    "Jupiter": "#ffa500",
    "Earth": "mediumseagreen",
}

plt.plot(states[:-44, 1], states[:-44, 2], lw=1.25, zorder=4)
plt.scatter([0], [0], c=colors["Sun"], s=25, marker="o", edgecolors="black", linewidths=0.5)

body_state_histories = []
body_states = []

for body in bodies:
    body_state_history = {
        epoch: spice.get_body_cartesian_position_at_epoch(body, "Sun", "J2000", "NONE", epoch)
        for epoch in state_keys
    }

    body_state_histories.append(body_state_history)
    body_states.append(result2array(body_state_history))


for i, body in enumerate(bodies):
    body_state_history = body_state_histories[i]
    body_state = np.array(body_states[i])

    if body == "Earth":
        pos1 = body_state_history[node_times[0]]
        plt.scatter(
            pos1[0],
            pos1[1],
            s=25,
            marker="o",
            c=colors[body],
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
        )

    if body == "Jupiter":
        pos1 = body_state_history[state_keys[420]]
        plt.scatter(
            pos1[0],
            pos1[1],
            s=30,
            marker="o",
            c=colors[body],
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
        )

    if body == "Neptune":
        pos1 = body_state_history[state_keys[node_time_indices[-1]]]
        plt.scatter(
            pos1[0],
            pos1[1],
            s=30,
            marker="o",
            c=colors[body],
            edgecolors="black",
            linewidths=0.5,
            zorder=7,
        )

    lims = {"Earth": 200, "Jupiter": 800, "Neptune": -1}
    plt.plot(
        body_state[: lims[body], 1],
        body_state[: lims[body], 2],
        color=colors[body],
        label=body,
        lw=1.25,
        alpha=0.8,
    )


plt.yticks(ticks=plt.yticks()[0], labels=[])
plt.xticks(ticks=plt.xticks()[0], labels=[])
plt.gca().set_aspect("equal")
plt.legend(fontsize=8, loc="upper left")

# plt.savefig("champion_trajectory.png", bbox_inches="tight", dpi=300)
# plt.savefig("champion_trajectory.pdf", bbox_inches="tight", dpi=300)

# %%
get_dates(x, run.p.number_of_legs)
