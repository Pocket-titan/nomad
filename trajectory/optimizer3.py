# %%
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pygmo as pg
import wat
import re
from pprint import pprint
from astropy.time import Time, TimeDelta

from tudatpy.trajectory_design import transfer_trajectory
from tudatpy.trajectory_design.transfer_trajectory import (
    unpowered_leg,
    dsm_velocity_based_leg,
    departure_node,
    swingby_node,
    capture_node,
)
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.util import result2array
from tudatpy import constants


np.set_printoptions(precision=2, suppress=True)
sns.set_theme(style="ticks", palette="Set2")
pal = sns.color_palette("Set2")


# %%
class Problem:
    def __init__(
        self,
        transfer_trajectory_object: transfer_trajectory.TransferTrajectory,
        bounds: np.ndarray,
        dim=1,
    ) -> None:
        self.transfer_trajectory_function = lambda: transfer_trajectory_object
        self.nn = transfer_trajectory_object.number_of_nodes
        self.nl = transfer_trajectory_object.number_of_legs

        nodes, legs = self._determine_leg_node_types()
        self.kinds = np.concatenate([legs, nodes])

        self.number_of_parameters = self.kinds[:, 2].astype(int).sum() + self.nn
        eps = np.finfo(np.float32).eps

        bound_map = {
            "leg": {
                "velocity-based DSM": [[0 + 10 * eps, 1 - 10 * eps]],
            },
            "node": {
                "escape_and_departure": [
                    [0, 1e10],
                    [0, 2 * np.pi],
                    [-np.pi / 2, np.pi / 2],
                ],
                "swingby": [
                    [0, 1e26],
                    [0, 2 * np.pi],
                    [0, 10_000],
                ],
            },
        }
        _bounds = np.zeros((self.number_of_parameters, 2))
        _bounds[: len(bounds)] = bounds
        j = self.nn
        for sort, kind, npar in self.kinds:
            npar = int(npar)

            if kind is None:
                j += npar
                continue

            if kind not in bound_map[sort]:
                raise Exception(
                    f"Add this {sort} kind to the bound_map: {kind}, {npar =}"
                )

            _bounds[j : j + npar] = bound_map[sort][kind]
            j += npar

        self.bounds = _bounds
        self.dim = dim

    def get_nobj(self) -> int:
        return self.dim

    def get_bounds(self) -> tuple[list[float]]:
        return self.bounds[:, 0], self.bounds[:, 1]

    def get_number_of_parameters(self) -> int:
        return self.number_of_parameters

    def fitness(self, x: list[float]) -> list[float]:
        trajectory = self.transfer_trajectory_function()
        node_times, leg_parameters, node_parameters = self.convert_parameters(x)

        time_of_flight = (node_times[-1] - node_times[0]) / constants.JULIAN_YEAR
        if time_of_flight > 25:
            return self._death_penalty()

        try:
            self.evaluate(node_times, leg_parameters, node_parameters)
        except Exception as e:
            print("Err:", e)
            print(node_times, leg_parameters, node_parameters)
            raise e

        try:
            self.evaluate(node_times, leg_parameters, node_parameters)
            delta_v = trajectory.delta_v
        except:  # noqa: E722
            return self._death_penalty()

        if self.dim == 1:
            return [delta_v]

        return [delta_v, time_of_flight]

    def evaluate(self, *args, **kwargs):
        return self.transfer_trajectory_function().evaluate(*args, **kwargs)

    def convert_parameters(self, trajectory_parameters):
        node_times = np.cumsum(trajectory_parameters[: self.nn])

        leg_parameters = []
        j = self.nn
        for n in self.kinds[self.kinds[:, 0] == "leg"][:, 2].astype(int):
            if n == 0:
                leg_parameters.append([])
            else:
                leg_parameters.append(trajectory_parameters[j : j + n])
                j += n

        node_parameters = []
        j = self.nn
        for n in self.kinds[self.kinds[:, 0] == "node"][:, 2].astype(int):
            if n == 0:
                node_parameters.append([])
            else:
                node_parameters.append(trajectory_parameters[j : j + n])
                j += n

        return node_times, leg_parameters, node_parameters

    @property
    def death_value(self) -> float:
        return 1e10

    def _death_penalty(self):
        return [self.death_value] * self.dim

    def _determine_leg_node_types(self) -> tuple[list[tuple[str, int, str]]]:
        nodes = [("node", None, 0)] * self.nn
        legs = [("leg", None, 0)] * self.nl

        def try_evaluate(*parameters):
            try:
                self.evaluate(*parameters)
            except RuntimeError as e:
                match = re.match(
                    r"Error when getting (?:leg|node) parameters for (leg|node) ([0-9]*) \(type: (.*)\).*\. (?:Leg|Node) should have ([0-9]*) free parameters",
                    str(e),
                )

                if match:
                    sort, idx, _type, npar = match.groups()

                    if sort == "leg":
                        arr = parameters[1]
                        kind = legs
                    if sort == "node":
                        arr = parameters[2]
                        kind = nodes

                    kind[int(idx)] = (sort, _type, int(npar))
                    arr[int(idx)].extend(
                        [0.5 for _ in range(int(npar) - len(arr[int(idx)]))]
                    )

                    return try_evaluate(*parameters)
                else:
                    raise Exception("Error parsing error message:", e)

            return parameters

        node_times = np.linspace(1, 1e3, num=self.nn)
        leg_free_parameters = [[] for _ in range(self.nl)]
        node_free_parameters = [[] for _ in range(self.nn)]

        try_evaluate(node_times, leg_free_parameters, node_free_parameters)

        return nodes, legs


# %%
central_body = "Sun"

transfer_body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = 3.8913e9
arrival_eccentricity = 0.999486

transfer_leg_settings, transfer_node_settings = (
    # transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
    #     transfer_body_order,
    #     departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    #     arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    # )
    transfer_trajectory.mga_settings_dsm_velocity_based_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
)

transfer_node_settings = [
    departure_node(departure_semi_major_axis, departure_eccentricity),
    swingby_node(3689000.0),
    swingby_node(600000000.0),
    capture_node(arrival_semi_major_axis, arrival_eccentricity),
]

transfer_leg_settings = [
    # unpowered_leg(),
    # unpowered_leg(),
    # unpowered_leg(),
    dsm_velocity_based_leg(),
    dsm_velocity_based_leg(),
    # unpowered_leg(),
    # unpowered_leg(),
    dsm_velocity_based_leg(),
]

# velocity -> unpowered is ok
# unpowered -> velocity is not ok

transfer_trajectory.print_parameter_definitions(
    transfer_leg_settings, transfer_node_settings
)

bodies = environment_setup.create_simplified_system_of_bodies()

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body,
)

node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]

departure_bounds = [[DateTime(2028, 4, 6).epoch(), DateTime(2033, 12, 31).epoch()]]
leg_tof_bounds = np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]]
leg_free_bounds = [
    [0.1, 0.99],
    [0.1, 0.99],
    [0.1, 0.99],
]
node_free_bounds = [
    [],
    [],
    [],
    # 1
    [],
    [],
    [],
    # 2
    [],
    [],
    [],
]
bounds = np.concatenate([departure_bounds, leg_tof_bounds])


p = Problem(
    transfer_trajectory_object,
    bounds,
    dim=2,
)


# %%
prob = pg.problem(p)

seed = 4444
pop_size = 20
neighbours = 5
num_generations = 10
num_evolutions = 5

algo = pg.algorithm(
    pg.moead(
        gen=num_generations,
        neighbours=neighbours,
        seed=seed,
    )
)

pop = pg.population(prob, size=pop_size, seed=seed)
algo.evolve(pop)


# %%

results = dict(f=[pop.get_f()], x=[pop.get_x()])

for i in range(num_generations):
    pop = algo.evolve(pop)

    results["f"].append(pop.get_f())
    results["x"].append(pop.get_x())

generations = [[i] * pop_size for i in range(num_generations + 1)]
fs = np.concatenate(results["f"])
xs = np.concatenate(results["x"])

# %%
df = pl.DataFrame(
    {
        "dv": fs[:, 0],
        "tof": fs[:, 1],
        "x": xs,
        "gen": np.concatenate(generations),
    }
).filter((pl.col("dv") != p.death_value) & (pl.col("tof") != p.death_value))

champions = (
    (
        df.group_by("gen")
        .agg(pl.all().sort_by("dv").head(5))
        .explode(pl.all().exclude("gen"))
    )
    .sort("dv")
    .filter(pl.col("gen") > 3)
)

plt.scatter(champions["tof"], champions["dv"], c=champions["gen"], cmap="viridis_r")

# %%
from contextlib import redirect_stdout
import io

f = io.StringIO()

with redirect_stdout(f):
    transfer_trajectory.print_parameter_definitions(
        transfer_leg_settings, transfer_node_settings
    )


s = f.getvalue()

# %%


# %%
