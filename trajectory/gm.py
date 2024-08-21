# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygmo as pg
import pykep as pk
import warnings
import wat
from pprint import pprint

from astropy.time import Time, TimeDelta
from astropy.constants import G

warnings.filterwarnings("ignore", module="erfa")
sns.set_theme(style="ticks", palette="Set2")

# %%
node_times = [
    *map(
        lambda x: Time(x, format="iso", precision=0),
        [
            "2031-02-09 00:00:00",
            "2031-05-15 00:00:00",
            "2033-01-03 00:00:00",
            "2047-02-13 00:00:00",
        ],
    )
]

planets = [
    *map(
        lambda x: pk.planet.jpl_lp(x.lower()),
        ["Earth", "Mars", "Jupiter", "Neptune"],
    )
]

t0 = [
    pk.epoch_from_string("2028-01-01 00:00:00"),
    pk.epoch_from_string("2033-01-01 00:00:00"),
]

tof = np.vectorize(lambda x: x.value)(np.diff(node_times))[:, None] * [
    [0.2, 3],
    [0.2, 3],
    [0.5, 1.25],
]

# tof = np.array([(node_times[-1] - node_times[0]).value]) * [0.4, 1]

# Going to Mars (does swingyby at Jupiter count?? idk), so:
vinfdep = 3

rN = pk.planet.jpl_lp("Neptune").osculating_elements(
    pk.epoch_from_string(node_times[-1].iso)
)[0]
rJ = pk.planet.jpl_lp("Jupiter").osculating_elements(
    pk.epoch_from_string(node_times[-2].iso)
)[0]
rE = pk.planet.jpl_lp("Earth").osculating_elements(
    pk.epoch_from_string(node_times[0].iso)
)[0]

vinfarr1 = np.abs(
    np.sqrt(2 * pk.MU_SUN / rN) * np.sqrt(rE / (rE + rN)) - np.sqrt(pk.MU_SUN / rN)
)
vinfarr2 = np.abs(
    np.sqrt(2 * pk.MU_SUN / rN) * np.sqrt(rJ / (rJ + rN)) - np.sqrt(pk.MU_SUN / rN)
)
vinfarr3 = 6.5  # oddysey

vinf = [vinfdep, vinfarr2]

rp = 2000e3
T = 213.5 * 24 * 3600
muN = pk.planet.jpl_lp("Neptune").mu_self
a = (muN * (T**2) / (4 * np.pi**2)) ** (1 / 3)
ra = 2 * a - rp
e = (ra - rp) / (ra + rp)

print(f"{a = :.2f}")
print(f"{e = :.8f}")


# %%
class Problem(pk.trajopt.mga):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fitness(self, x):
        try:
            return super().fitness(x)
        except:
            return [1e10]


udp = Problem(
    seq=planets,
    t0=t0,
    tof=tof,
    vinf=vinfdep,
    multi_objective=False,
    tof_encoding="direct",
    # orbit_insertion=True,
    # e_target=e,
    # rp_target=rp,
)

prob = pg.problem(udp)
prob.c_tol = 1e-4
print(prob)

# %%
uda = pg.sade(gen=100)
algo = pg.algorithm(uda)

pop = pg.population(prob, 300)

number_of_evolutions = 200

for i in range(number_of_evolutions):
    pop = algo.evolve(pop)


# %%
dv = pop.champion_f.item()
times = np.cumsum(pop.champion_x)

J2000 = Time("J2000", format="jyear_str")
times = J2000 + TimeDelta(times, format="jd")

print(f"dv: {dv:.2f} m/s")
print("Dates:")
pprint([time.iso for time in times])

# %%
udp.pretty(pop.champion_x)
udp.plot(pop.champion_x)
