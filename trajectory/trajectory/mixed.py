# %%
import numpy as np
from tudatpy.trajectory_design import transfer_trajectory, shape_based_thrust
from tudatpy.numerical_simulation import environment_setup

# # Define the order of bodies (nodes) for gravity assists
# transfer_body_order = ["Earth", "Venus", "Venus", "Earth", "Jupiter", "Saturn"]

# no_of_nodes = len(transfer_body_order)

# # define ToF values per leg
# time_of_flight = np.array(
#     [1.97844702e07, 6.68197546e07, 1.00154317e07, 3.2306349e8, 4.21392438e8]
# )  # s

# # define number of revolutions per leg
# number_of_revolutions = np.array([2, 0, 1, 1, 0])

# # Define the departure and insertion orbits
# departure_semi_major_axis = np.inf
# departure_eccentricity = 0.0
# arrival_semi_major_axis = 1.0895e8 / 0.02
# arrival_eccentricity = 0.98

# # Determine number of legs and GA's
# no_of_legs = len(transfer_body_order) - 1
# no_of_gas = len(transfer_body_order) - 2

# # Create transfer leg settings
# transfer_leg_settings = []
# for i in range(no_of_legs):
#     radial_velocity_functions = shape_based_thrust.recommended_radial_hodograph_functions(
#         time_of_flight[i]
#     )
#     normal_velocity_functions = shape_based_thrust.recommended_normal_hodograph_functions(
#         time_of_flight[i]
#     )
#     axial_velocity_functions = shape_based_thrust.recommended_axial_hodograph_functions(
#         time_of_flight[i],
#         number_of_revolutions[i],
#     )
#     transfer_leg_settings.append(
#         transfer_trajectory.hodographic_shaping_leg(
#             radial_velocity_functions, normal_velocity_functions, axial_velocity_functions
#         )
#     )

# # Create transfer node settings
# transfer_node_settings = []
# transfer_node_settings.append(
#     transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity)
# )
# for i in range(no_of_gas):
#     transfer_node_settings.append(transfer_trajectory.swingby_node())

# transfer_node_settings.append(
#     transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity)
# )

# # Create transfer trajectory
# central_body = "Sun"

# bodies = environment_setup.create_simplified_system_of_bodies()

# transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
#     bodies,
#     transfer_leg_settings,
#     transfer_node_settings,
#     transfer_body_order,
#     central_body,
# )


# transfer_trajectory.print_parameter_definitions(
#     leg_settings=transfer_leg_settings, node_settings=transfer_node_settings
# )

# transfer_trajectory_object.evaluate(np.linspace(0, 1e9, no_of_nodes),
#                                     [[0], [0], [0], [0], [0]], np.)

# %%
central_body = "Sun"

transfer_body_order = ["Earth", "Venus", "Earth", "Mars", "Jupiter"]

bodies = environment_setup.create_simplified_system_of_bodies()

num_nodes = len(transfer_body_order)
num_legs = num_nodes - 1

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0
arrival_semi_major_axis = 1.0895e8 / 0.02
arrival_eccentricity = 0.98

transfer_node_settings = [
    transfer_trajectory.departure_node(departure_semi_major_axis, departure_eccentricity),
    transfer_trajectory.swingby_node(600_000),
    transfer_trajectory.swingby_node(600_000),
    transfer_trajectory.swingby_node(600_000),
    transfer_trajectory.capture_node(arrival_semi_major_axis, arrival_eccentricity),
]

transfer_leg_settings = [
    transfer_trajectory.unpowered_leg(),
    transfer_trajectory.dsm_velocity_based_leg(),
    transfer_trajectory.unpowered_leg(),
    transfer_trajectory.dsm_velocity_based_leg(),
]

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body,
)

transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

transfer_trajectory_object.evaluate(
    [0, 1e8, 9e8, 3e9, 4e9],
    [[], [0.5], [], [0.5]],
    [[], [700_000, 0, 10], [], [2e6, 0, 1000], []],
)

(
    transfer_trajectory_object.delta_v_per_leg,
    transfer_trajectory_object.delta_v_per_node,
    transfer_trajectory_object.delta_v,
)

# %%
