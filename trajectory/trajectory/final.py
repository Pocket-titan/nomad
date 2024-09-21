# %%
from pathlib import Path
from tudatpy.astro.time_conversion import DateTime
from tudatpy.interface import spice
from tudatpy import constants
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
from wishlist import create_unpowered, create_1dsm

sns.set_theme(style="ticks", palette="deep")

kernels = Path(__file__).parents[1] / "spice"

spice.load_standard_kernels()
# spice.load_kernel(str((kernels / "de440s.bsp").absolute()))
spice.load_kernel(str((kernels / "nep097.bsp").absolute()))
# spice.load_kernel(str((kernels / "jup344-s2003_j24.bsp").absolute()))
# spice.load_kernel(str((kernels / "mar097.bsp").absolute()))

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
df1 = (
    pl.scan_parquet(FOLDER / "trajectories_1dsm.parquet")
    .filter(pl.col("traj_dv") <= 10_000)
    .collect()
)

runs1 = read_runs(FOLDER / "neptune_1dsm")
run_names1 = [x.name for x in runs1]

# %%


# %%


def map_batches1(x, body_order, dsm_leg_index, run_name):
    run = runs1[run_names1.index(run_name[0])]

    obj, _, _ = create_1dsm(
        body_order=[body_map[x] for x in [*body_order[0]]],
        departure_semi_major_axis=a_dep,
        departure_eccentricity=e_dep,
        arrival_semi_major_axis=a_arr,
        arrival_eccentricity=e_arr,
        dsm_leg_index=dsm_leg_index[0],
    )

    dvs = []
    for x in x:
        try:
            obj.evaluate(*run.p.convert_parameters(x.to_numpy()))
            dv = obj.delta_v
        except Exception:
            dv = np.nan
        dvs.append(dv)

    return pl.Series(dvs)


df1 = (
    df1
    # .filter(pl.col("gen") > 50)
    # .group_by(["body_order", "dsm_leg_index"])
    # .agg(pl.all().head(10))
    # .explode(pl.exclude(["body_order", "dsm_leg_index"]))
    .with_columns(
        pl.struct(["x", "body_order", "dsm_leg_index", "run_name"])
        .map_batches(
            lambda x: map_batches1(
                x.struct.field("x"),
                x.struct.field("body_order"),
                x.struct.field("dsm_leg_index"),
                x.struct.field("run_name"),
            )
        )
        .over(["body_order", "dsm_leg_index"])
        .alias("traj_dv")
    ).select(
        [
            "traj_dv",
            "tof",
            "x",
            "body_order",
            "gen",
            "departure",
            "arrival",
            "kind",
            "algo",
            "dsm_leg_index",
            "run_name",
            "dv",
        ]
    )
)

# %%
df1.to_pandas().to_parquet(FOLDER / "trajectories_1dsm_2.parquet", compression="lz4")

# %%
df2 = pl.scan_parquet(FOLDER / "trajectories_unpowered.parquet").filter(
    pl.col("gen") > 10
)

runs2 = read_runs(FOLDER / "neptune_unpowered")
run_names2 = [x.name for x in runs2]


def map_batches2(x, body_order, run_name):
    run = runs2[run_names2.index(run_name[0])]

    obj, _, _ = create_unpowered(
        body_order=[body_map[x] for x in [*body_order[0]]],
        departure_semi_major_axis=a_dep,
        departure_eccentricity=e_dep,
        arrival_semi_major_axis=a_arr,
        arrival_eccentricity=e_arr,
    )

    dvs = []
    for x in x:
        try:
            obj.evaluate(*run.p.convert_parameters(x.to_numpy()))
            dv = obj.delta_v
        except Exception:
            dv = np.nan
        dvs.append(dv)

    return pl.Series(dvs)


df2 = (
    df2
    # .filter(pl.col("gen") > 50)
    # .group_by(["body_order", "dsm_leg_index"])
    # .agg(pl.all().head(10))
    # .explode(pl.exclude(["body_order", "dsm_leg_index"]))
    .with_columns(
        pl.struct(["x", "body_order", "run_name"])
        .map_batches(
            lambda x: map_batches2(
                x.struct.field("x"),
                x.struct.field("body_order"),
                x.struct.field("run_name"),
            )
        )
        .over(["body_order"])
        .alias("traj_dv")
    ).select(
        [
            "traj_dv",
            "tof",
            "x",
            "body_order",
            "gen",
            "departure",
            "arrival",
            "kind",
            "algo",
            "dsm_leg_index",
            "run_name",
            "dv",
        ]
    )
)

# %%
df2.to_pandas().to_parquet(FOLDER / "trajectories_unpowered_2.parquet", compression="lz4")


# %%
df = pl.concat([df1, df2]).collect()

# %%
# df.collect().write_parquet(FOLDER / "trajectories.parquet", compression="lz4")
# %%
df.collect().to_pandas().to_parquet(FOLDER / "trajectories.parquet", compression="lz4")

# %%
run = runs[0]

dsm_leg_index = int(run.name.split("_1dsm_")[1].split("_")[0])

a_arr = 1.9203e9
e_arr = 0.999999

r_dep = 185e3 + R_EARTH
a_dep = r_dep
e_dep = 0.0

obj, _, _ = create_1dsm(
    body_order=run.body_order,
    departure_semi_major_axis=a_dep,
    departure_eccentricity=e_dep,
    arrival_semi_major_axis=a_arr,
    arrival_eccentricity=e_arr,
    dsm_leg_index=dsm_leg_index,
)

champion = run.df.sort("dv").head(1).collect()[0]
champion_x = champion["x"].item().to_numpy()

node_times, leg_parameters, node_parameters = run.p.convert_parameters(champion_x)
obj.evaluate(node_times, leg_parameters, node_parameters)
state_history = obj.states_along_trajectory(500)
states = result2array(state_history)

x, y, z, vx, vy, vz = states[:, 1:].T

r = np.sqrt(x**2 + y**2 + z**2)
v = np.sqrt(vx**2 + vy**2 + vz**2)

eps = np.abs(1 / 2 * v**2 - MU_SUN / r)
C3 = 2 * eps  # * 1e-6
vinf = np.sqrt(C3)

print(vinf[-1], obj.delta_v, obj.delta_v_per_node)

# %%
v_earth = np.sqrt(
    (
        spice.get_body_cartesian_state_at_epoch(
            "Earth", "Sun", "J2000", "NONE", node_times[0]
        )[3:]
        ** 2
    ).sum()
)

v_neptune = np.sqrt(
    (
        spice.get_body_cartesian_state_at_epoch(
            "Neptune", "Sun", "J2000", "NONE", node_times[-1]
        )[3:]
        ** 2
    ).sum()
)


def vescape_earth(r):
    return np.sqrt(2 * MU_EARTH / r)


print(f"v_earth = {v_earth * 1e-3:.2f} km/s")
print(f"v_neptune = {v_neptune * 1e-3:.2f} km/s")
print(f"vescape_earth = {vescape_earth(R_EARTH) * 1e-3:.2f} km/s")

# %%
xe, ye, ze = result2array(
    {
        epoch: spice.get_body_cartesian_state_at_epoch(
            "Earth", "Sun", "J2000", "NONE", epoch
        )[:3]
        for epoch in state_history.keys()
    }
)[:, 1:].T

plt.plot(xe[:10], ye[:10], label="Earth")
plt.plot(x[:10], y[:10], label="Trajectory")
plt.scatter([0], [0])

dx, dy, dz = (x - xe), (y - ye), (z - ze)
dr = np.sqrt(dx**2 + dy**2 + dz**2)  # dist from earth

# plt.plot([xe[0], x[0]], [ye[0], y[0]], label="Departure")
plt.plot([xe[0], xe[0] + dx[0]], [ye[0], ye[0] + dy[0]], label="Departure")

# plt.plot(x, y)

# %%


# %%
np.sqrt((vinf[0] * 1e3) ** 2 + 2 * MU_EARTH / (R_EARTH + 300e3)) / 1e3

plt.plot(v / 1e3)

# %%
dv_frompark = 6206.018912915056
dv_frominf = 8631.157878722135

ddv = dv_frominf - dv_frompark
ddv

# %%
vcirc = np.sqrt(MU_EARTH / r_dep)

eps2 = 0.5 * C3[0]
eps1 = 0.5 * vcirc**2 - MU_EARTH / r_dep

eps1, eps2
# %%
vsoi = np.sqrt(
    # vescape_earth(dr[0]) ** 2 + C3[0]
    v_earth**2 + C3[0]
)  # velocity at edge of SOI so we can get onto hyperbolic transfer trajectory

eps_soi = 0.5 * vsoi**2 - MU_EARTH / dr[0]

vp = np.sqrt(2 * (eps_soi + MU_EARTH / r_dep))  # v_periapsis
dv0 = np.abs(vp - vcirc)
dv0 / 1e3

# %%
vinf_dep = np.abs(v[0] - v_earth)

v_park = np.sqrt(MU_EARTH / r_dep)  # v_circ

vp = np.sqrt(vinf_dep**2 + 2 * MU_EARTH / r_dep)
dv0 = np.abs(vp - v_park)
dv0
# %%
