# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygmo as pg
import wat
import re
from pprint import pprint
from astropy.time import Time, TimeDelta

from tudatpy.trajectory_design import transfer_trajectory
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy import constants
from tudatpy.util import result2array

np.set_printoptions(precision=2, suppress=True)
sns.set_theme(style="ticks", palette="Set2")
pal = sns.color_palette("Set2")


# %%
def convert_trajectory_parameters(
    transfer_trajectory_object,
    trajectory_parameters,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    nn = transfer_trajectory_object.number_of_nodes
    nl = transfer_trajectory_object.number_of_legs

    node_times = np.cumsum(trajectory_parameters[0:nn])

    leg_params = np.reshape(trajectory_parameters[nn : (nn + nl)], (-1, 1)).tolist()
    leg_params[len(leg_params) : nl] = [[] for _ in range(nl - len(leg_params))]

    node_params = np.reshape(trajectory_parameters[(nn + nl) :], (-1, 3)).tolist()
    node_params[len(node_params) : nn] = [[] for _ in range(nn - len(node_params))]

    return node_times, leg_params, node_params


class Problem:
    def __init__(
        self,
        transfer_trajectory_object,
        departure_date_lb: float,  # Lower bound on departure date
        departure_date_ub: float,  # Upper bound on departure date
        legs_tof_lb: np.ndarray,  # Lower bounds of each leg's time of flight
        legs_tof_ub: np.ndarray,  # Upper bounds of each leg's time of flight,
        bounds: np.ndarray = None,
        dim=1,
    ) -> None:
        self.departure_date_lb = departure_date_lb
        self.departure_date_ub = departure_date_ub
        self.legs_tof_lb = legs_tof_lb
        self.legs_tof_ub = legs_tof_ub
        self.bounds = bounds
        self.dim = dim

        # Save the transfer trajectory object as a lambda function
        # PyGMO internally pickles its user defined objects and some objects cannot be pickled properly without using lambda functions.
        self.transfer_trajectory_function = lambda: transfer_trajectory_object

    def get_nobj(self) -> int:
        return self.dim

    def get_bounds(self) -> tuple:
        """
        Returns the boundaries of the decision variables.
        """
        transfer_trajectory_obj = self.transfer_trajectory_function()
        number_of_parameters = self.get_number_of_parameters()
        nl = transfer_trajectory_obj.number_of_legs

        maxval = np.finfo(float).max
        bounds = np.ones((number_of_parameters, 2)) * [[-maxval, maxval]]
        bounds[nl + 1 : 2 * nl + 1, 0] = 0

        bounds[0] = [self.departure_date_lb, self.departure_date_ub]
        bounds[1 : nl + 1] = np.array([self.legs_tof_lb, self.legs_tof_ub]).T
        if self.bounds is not None:
            bounds[nl + 1 :] = self.bounds

        return (bounds[:, 0], bounds[:, 1])

    def get_number_of_parameters(self):
        transfer_trajectory_obj = self.transfer_trajectory_function()

        number_of_parameters = (
            transfer_trajectory_obj.number_of_nodes
            + transfer_trajectory_obj.number_of_legs * 1
            + (transfer_trajectory_obj.number_of_nodes - 1) * 3
        )

        return number_of_parameters

    def fitness(self, trajectory_parameters: list[float]) -> list:
        """
        Returns delta V of the transfer trajectory object with the given set of trajectory parameters
        """

        trajectory = self.transfer_trajectory_function()
        node_times, leg_free_parameters, node_free_parameters = (
            convert_trajectory_parameters(trajectory, trajectory_parameters)
        )

        death_penalty = 1e10

        time_of_flight = (node_times[-1] - node_times[0]) / constants.JULIAN_YEAR
        if time_of_flight > 20:  # 20 year TOF constraint?
            return [death_penalty] * self.dim

        try:
            trajectory.evaluate(node_times, leg_free_parameters, node_free_parameters)
            delta_v = trajectory.delta_v
        # If there was some error in the evaluation of the trajectory, use a very large deltaV as penalty
        except:  # noqa: E722
            return [death_penalty] * self.dim

        if self.dim == 1:
            return [delta_v]

        return [delta_v, time_of_flight]


# %%
central_body = "Sun"

transfer_body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = 3.8913e9
arrival_eccentricity = 0.999486

transfer_leg_settings, transfer_node_settings = (
    transfer_trajectory.mga_settings_dsm_velocity_based_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
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

transfer_trajectory_object.evaluate(
    node_times,
    [[0], [0], [0]],
    [[1000, 1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000], []],
)

# odyssey_dv = transfer_trajectory_object.delta_v

odyssey_dv = 10_000

departure_date_lb = DateTime(2028, 4, 6).epoch()
departure_date_ub = DateTime(2033, 12, 31).epoch()

leg_bounds = np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]]

legs_tof_lb = leg_bounds[:, 0]
legs_tof_ub = leg_bounds[:, 1]

bounds = np.concatenate([[[departure_date_lb, departure_date_ub]], leg_bounds])

optimizer = Problem(
    transfer_trajectory_object,
    departure_date_lb,
    departure_date_ub,
    legs_tof_lb,
    legs_tof_ub,
    bounds=np.array(
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 3000e3],
            [0, 2 * np.pi],
            [0, 2 * np.pi],
            [0, 1e20],
            [0, 2 * np.pi],
            [0, 3000e3],
            [0, 1e20],
            [0, 2 * np.pi],
            [0, 3000e3],
        ]
    ),
    dim=2,
)

# %%

prob = pg.problem(optimizer)

# number_of_generations = 1
number_of_generations = 1
optimization_seed = 4444
neighbours = 10
population_size = 20
number_of_evolutions = 1000

algo = pg.algorithm(
    pg.moead(
        gen=number_of_generations,
        neighbours=neighbours,
        seed=optimization_seed,
    )
)
# algo = pg.algorithm(pg.de(gen=number_of_generations, seed=optimization_seed, F=0.5))

# population_size = 20

pop = pg.population(prob, size=population_size, seed=optimization_seed)

# number_of_evolutions = 800

pop = algo.evolve(pop)
# %%
fs = []
xs = []

for i in range(number_of_evolutions):
    pop = algo.evolve(pop)

    fs.extend(pop.get_f())
    xs.extend(pop.get_x())

print("The optimization has finished")

fs = np.array(fs)
xs = np.array(xs)

sfs = np.split(fs, len(fs) // population_size)
sxs = np.split(xs, len(xs) // population_size)

# %%
for i in range(len(sfs)):
    plt.scatter(
        sfs[i][:, 1],
        sfs[i][:, 0],
        color=pal[0],
        alpha=0.1 + (i / len(sfs)) ** 2 * 0.9,
    )
# plt.scatter(fs[:, 1], fs[:, 0])

plt.ylim([0, 50_000])

# %%
dv = pop.champion_f.item()
times = np.cumsum(pop.champion_x)

J2000 = Time("J2000", format="jyear_str")
times = J2000 + TimeDelta(times, format="sec")

print(f"dv: {dv:.2f} m/s")
print(f"Diff w oddysey dv: {dv - odyssey_dv:.2f} m/s")
print("Dates:")
pprint([time.iso for time in times])

# Reevaluate the transfer trajectory using the champion design variables
node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(
    transfer_trajectory_object, pop.champion_x
)
transfer_trajectory_object.evaluate(
    node_times, leg_free_parameters, node_free_parameters
)

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, zorder=0)
ax.scatter(
    fly_by_states[0, 0] / au,
    fly_by_states[0, 1] / au,
    label="Earth departure",
)
ax.scatter(
    fly_by_states[1, 0] / au,
    fly_by_states[1, 1] / au,
    label="Mars fly-by",
)
ax.scatter(
    fly_by_states[2, 0] / au,
    fly_by_states[2, 1] / au,
    label="Jupiter swing-by",
)
ax.scatter(
    fly_by_states[3, 0] / au,
    fly_by_states[3, 1] / au,
    label="Neptune arrival",
)
ax.scatter([0], [0], color="orange", label="Sun")
ax.set_xlabel("x wrt Sun [AU]")
ax.set_ylabel("y wrt Sun [AU]")
ax.set_aspect("equal")
ax.legend(bbox_to_anchor=[1, 1])


# %%
class _Problem:
    def __init__(
        self,
        transfer_trajectory_object,
        bounds: np.ndarray,
        dim=1,
    ) -> None:
        self.transfer_trajectory_function = lambda: transfer_trajectory_object
        self.nn = transfer_trajectory_object.number_of_nodes
        self.nl = transfer_trajectory_object.number_of_legs

        nodes, legs = self._determine_leg_node_types()
        self.kinds = np.concatenate([legs, nodes])

        self.number_of_parameters = self.kinds[:, 2].astype(int).sum() + self.nn

        # There is a free parameter for each leg, representing the leg’s time-of-flight fraction at which the DSM takes place

        # For the departure node:
        # 1. Magnitude of the relative velocity w.r.t. the departure planet after departure.
        # 2. In-plane angle of the relative velocity w.r.t. the departure planet after departure.
        # 3. Out-of-plane angle of the relative velocity w.r.t. the departure planet after departure.

        # For the swing-by nodes:
        # 1. Periapsis radius.
        # 2. Rotation angle.
        # 3. Magnitude of DV applied at periapsis.

        bound_map = {
            "leg": {
                "velocity-based DSM": [[0, 1]],
            },
            "node": {
                "escape_and_departure": [
                    [0, 1e10],
                    [0, 2 * np.pi],
                    [0, 2 * np.pi],
                ],
                "swingby": [
                    [0, 1e26],
                    [0, 2 * np.pi],
                    [0, 1e10],
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
        if time_of_flight > 20:
            return self._death_penalty()

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


departure_date_lb = DateTime(2028, 4, 6).epoch()
departure_date_ub = DateTime(2033, 12, 31).epoch()

leg_bounds = np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]]

p = _Problem(
    transfer_trajectory_object,
    leg_bounds,
)

# %%


# %%
prob = pg.problem(p)

number_of_generations = 1
optimization_seed = 4444
neighbours = 10
population_size = 20
number_of_evolutions = 1000

algo = pg.algorithm(
    pg.moead(
        gen=number_of_generations,
        neighbours=neighbours,
        seed=optimization_seed,
    )
)

pop = pg.population(prob, size=population_size, seed=optimization_seed)

pop = algo.evolve(pop)

# %%
