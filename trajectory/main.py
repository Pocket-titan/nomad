# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.util import result2array

sns.set_theme(
    style="ticks",
    palette="Set2",
)

# %%
central_body = "Sun"

transfer_body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = 1.0895e8 / 0.02
arrival_eccentricity = 0.98

# %%
transfer_leg_settings, transfer_node_settings = (
    transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
)

# %%
bodies = environment_setup.create_simplified_system_of_bodies()

# %%
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
leg_free_parameters = [list() for _ in node_times]
node_free_parameters = [list() for _ in node_times]

transfer_trajectory_object.evaluate(
    node_times, leg_free_parameters, node_free_parameters
)

# %%
print(f"dv: {transfer_trajectory_object.delta_v:.2f}")

state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

# %%
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
