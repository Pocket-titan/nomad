# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygmo as pg
import wat
from pprint import pprint
from astropy.time import Time, TimeDelta

from tudatpy.trajectory_design import transfer_trajectory
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy import constants
from tudatpy.util import result2array

sns.set_theme(style="ticks", palette="Set2")
pal = sns.color_palette("Set2")


# %%
def convert_trajectory_parameters(
    transfer_trajectory_object,
    trajectory_parameters,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    # Declare lists of transfer parameters
    node_times = []
    leg_free_parameters = []
    node_free_parameters = []

    # Extract from trajectory parameters the lists with each type of parameters
    departure_time = trajectory_parameters[0]
    times_of_flight_per_leg = trajectory_parameters[1:]

    # Get node times
    # Node time for the intial node: departure time
    node_times.append(departure_time)
    # None times for other nodes: node time of the previous node plus time of flight
    accumulated_time = departure_time
    for i in range(0, transfer_trajectory_object.number_of_nodes - 1):
        accumulated_time += times_of_flight_per_leg[i]
        node_times.append(accumulated_time)

    # Get leg free parameters and node free parameters: one empty list per leg
    for i in range(transfer_trajectory_object.number_of_legs):
        leg_free_parameters.append([])
    # One empty array for each node
    for i in range(transfer_trajectory_object.number_of_nodes):
        node_free_parameters.append([])

    return node_times, leg_free_parameters, node_free_parameters


class Problem:
    def __init__(
        self,
        transfer_trajectory_object,
        departure_date_lb: float,  # Lower bound on departure date
        departure_date_ub: float,  # Upper bound on departure date
        legs_tof_lb: np.ndarray,  # Lower bounds of each leg's time of flight
        legs_tof_ub: np.ndarray,  # Upper bounds of each leg's time of flight,
        dim=1,
    ) -> None:
        self.departure_date_lb = departure_date_lb
        self.departure_date_ub = departure_date_ub
        self.legs_tof_lb = legs_tof_lb
        self.legs_tof_ub = legs_tof_ub
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

        lower_bound = list(np.empty(number_of_parameters))
        upper_bound = list(np.empty(number_of_parameters))

        lower_bound[0] = self.departure_date_lb
        upper_bound[0] = self.departure_date_ub

        # Define boundaries on time of flight between bodies ['Earth', 'Venus', 'Venus', 'Earth', 'Jupiter', 'Saturn']
        for i in range(0, transfer_trajectory_obj.number_of_legs):
            lower_bound[i + 1] = self.legs_tof_lb[i]
            upper_bound[i + 1] = self.legs_tof_ub[i]

        bounds = (lower_bound, upper_bound)
        return bounds

    def get_number_of_parameters(self):
        transfer_trajectory_obj = self.transfer_trajectory_function()
        # Get number of parameters: it's the number of nodes (time at the first node, and time of flight to reach each subsequent node)
        number_of_parameters = transfer_trajectory_obj.number_of_nodes
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

# arrival_semi_major_axis = 1.0895e8 / 0.02
# arrival_eccentricity = 0.98

arrival_semi_major_axis = 3.8913e9
arrival_eccentricity = 0.999486

# dv of Uranus upper stage = 2708 m/s

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

bodies = environment_setup.create_simplified_system_of_bodies()

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body,
)

# %%

node_times = [
    DateTime(2031, 2, 9).epoch(),
    DateTime(2031, 5, 15).epoch(),
    DateTime(2033, 1, 3).epoch(),
    DateTime(2047, 2, 13).epoch(),
]

# transfer_trajectory_object.evaluate(
#     node_times,
#     [],
#     [[1000, 1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000], []],
# )

# odyssey_dv = transfer_trajectory_object.delta_v

odyssey_dv = 10_000

departure_date_lb = DateTime(2028, 4, 6).epoch()
departure_date_ub = DateTime(2033, 12, 31).epoch()

leg_bounds = np.diff(node_times)[:, None] * [[0.2, 3], [0.2, 3], [0.5, 1.25]]

legs_tof_lb = leg_bounds[:, 0]
legs_tof_ub = leg_bounds[:, 1]

optimizer = Problem(
    transfer_trajectory_object,
    departure_date_lb,
    departure_date_ub,
    legs_tof_lb,
    legs_tof_ub,
    dim=2,
)

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
transfer_trajectory_object.time_of_flight / constants.JULIAN_YEAR
