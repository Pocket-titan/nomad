# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from godot.core import tempo, util, constants
from godot.model import common
from godot import cosmos
import pygmo as pg

folder = os.path.dirname(__file__)

# %%

util.suppressLogger()
uni_config = cosmos.util.load_yaml(os.path.join(folder, "universe.yml"))
uni = cosmos.Universe(uni_config)


tra_config = cosmos.util.load_yaml(os.path.join(folder, "trajectory.yml"))
tra = cosmos.Trajectory(uni, tra_config)


def man_data(name):
    man = tra.getManoeuvreBook()[name]
    start = tra.point(man.start)
    end = tra.point(man.end)
    ran = tempo.EpochRange(start, end)
    grid = ran.createGrid(1.0 * tempo.SecondsInDay)
    mass = uni.evaluables.get("SC_mass")
    t = [e.mjd() for e in grid]
    r = np.asarray([uni.frames.vector3("Sun", "SC_center", "EMC", e) / constants.AU for e in grid])
    man.model.setActive(True)
    thr = np.asarray([np.linalg.norm(man.model.eval(e)) * mass.eval(e) for e in grid])
    man.model.setActive(False)
    return t, r, thr


def plot():
    tra.compute(False)
    ran = tempo.EpochRange(tempo.Epoch("2020-01-01 TDB"), tempo.Epoch("2022-01-01 TDB"))
    earth = np.asarray(
        [
            uni.frames.vector3("Sun", "Earth", "EMC", e) / constants.AU
            for e in ran.createGrid(1.0 * tempo.SecondsInDay)
        ]
    )
    mars = np.asarray(
        [
            uni.frames.vector3("Sun", "Mars", "EMC", e) / constants.AU
            for e in ran.createGrid(1.0 * tempo.SecondsInDay)
        ]
    )

    def pos(e):
        try:
            return uni.frames.vector3("Sun", "SC_center", "EMC", e) / constants.AU
        except:
            return [np.nan] * 3

    grid = tra.range().createGrid(1.0 * tempo.SecondsInDay)
    t = [e.mjd() for e in grid]
    thr = np.asarray([uni.evaluables.get("SC_sep_thrust").eval(e) for e in grid])
    x = np.asarray([pos(e) for e in grid])

    man1_t, man1_r, man1_thr = man_data("man1")
    man2_t, man2_r, man2_thr = man_data("man2")

    match1_ep = tra.point("match_inter_left")
    match1_val = uni.evaluables.get("SC_sep_thrust").eval(match1_ep)
    match2_ep = tra.point("match_inter_right")
    match2_val = uni.evaluables.get("SC_sep_thrust").eval(match2_ep)

    plt.figure(figsize=(8, 8))
    plt.xlabel("EMC X (AU)")
    plt.ylabel("EMC Y (AU)")
    plt.plot(earth[:, 0], earth[:, 1], "--", label="Earth")
    plt.plot(mars[:, 0], mars[:, 1], "--", label="Mars")
    plt.plot(x[:, 0], x[:, 1], "-k", linewidth=2, label="Transfer")
    plt.plot(man1_r[:, 0], man1_r[:, 1], "-r", linewidth=4, label="SEP 1")
    plt.plot(man2_r[:, 0], man2_r[:, 1], "-g", linewidth=4, label="SEP 2")
    plt.plot(x[0, 0], x[0, 1], "ok", ms=10)
    plt.plot(x[-1, 0], x[-1, 1], "ok", ms=10)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend()
    plt.grid()

    plt.figure(figsize=(8, 4))
    plt.xlabel("Time (MJD)")
    plt.ylabel("SEP thrust (mN)")
    plt.plot(t, 1e6 * thr, "--k", linewidth=1, label="Max thrust")
    plt.plot([tra.range().start().mjd()], 1e6 * thr[0], "o", ms=10, label="Earth")
    plt.plot([tra.range().end().mjd()], 1e6 * thr[-1], "o", ms=10, label="Mars")
    plt.plot([match1_ep.mjd()], 1e6 * match1_val, "dk", ms=5, label="Match")
    plt.plot([match2_ep.mjd()], 1e6 * match2_val, "dk", ms=5)
    plt.plot(man1_t, 1e6 * man1_thr, "-r", linewidth=3, label="SEP 1")
    plt.plot(man2_t, 1e6 * man2_thr, "-g", linewidth=3, label="SEP 2")
    plt.xlim([tra.range().start().mjd(), tra.range().end().mjd()])
    plt.ylim([0, 1000])
    plt.legend()
    plt.grid()


plot()


def summary():
    print(
        f"Escape velocity: {uni.parameters.get('Departure_SC_center_vin').getPhysicalValue():.3f} km/s"
    )
    print(f"Launch mass:     {uni.parameters.get('Departure_SC_mass').getPhysicalValue():.1f} kg")
    print(f"Arrival mass:    {uni.parameters.get('Arrival_SC_mass').getPhysicalValue():.1f} kg")
    print(f"Total delta-V:   {uni.parameters.get('Arrival_SC_dv').getPhysicalValue():.3f} km/s")


pro_config = cosmos.util.load_yaml(os.path.join(folder, "problem.yml"))
pro = cosmos.Problem(uni, [tra], pro_config)
problem = pg.problem(pro)
conTol = 1e-6
problem.c_tol = [conTol] * problem.get_nc()
x0 = pro.get_x()
pop = pg.population(problem, 0)
pop.push_back(x0)
algo = pg.algorithm(pg.nlopt("slsqp"))
algo.set_verbosity(1)
pop = algo.evolve(pop)


plot()
summary()

# %%
