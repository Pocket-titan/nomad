# %%
from pathlib import Path
from astropy.time import Time, TimeDelta
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np

from trajectory.lib.run import Run

folder = Path(__file__).parents[1] / "runs_unpowered_1" / "runs"
print(folder)

# %%
run_names = [x.name for x in folder.iterdir() if x.is_dir()]

run = Run.read(folder, run_names[2])
run

# %%
champions = run.df.sort("dv").head(500_000).collect()
champions = champions.reverse()
xs = np.array(champions.select(pl.col("x")).to_series().to_list())

J2000 = Time("J2000", format="jyear_str")

departures = J2000 + TimeDelta(xs[:, 0], format="sec")
arrivals = J2000 + TimeDelta(
    np.sum(xs[:, : run.p.number_of_legs + 1], axis=-1), format="sec"
)

dvs = champions["dv"]

plt.title(f"{', '.join(run.body_order)}. Unpowered")
plt.scatter(
    departures.to_value("datetime64"),
    arrivals.to_value("datetime64"),
    cmap="viridis_r",
    c=dvs,
)
plt.colorbar(
    label=r"$\Delta$V [km/s]",
    format=ticker.FuncFormatter(lambda x, _: f"{x / 1000:.2f}"),
)
plt.xticks(rotation=-45)
plt.xlabel("Departure date")
plt.ylabel("Arrival date")

# %%
