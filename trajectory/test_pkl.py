# %%
import numpy as np
import cloudpickle as pkl
from tudatpy.numerical_simulation import environment_setup
from tudatpy.trajectory_design.transfer_trajectory import (
    TransferTrajectory,
    TransferLegSettings,
    TransferNodeSettings,
    print_parameter_definitions,
    create_transfer_trajectory,
    mga_settings_unpowered_unperturbed_legs,
)
import wat


central_body = "Sun"

body_order = ["Earth", "Mars", "Jupiter", "Neptune"]

departure_semi_major_axis = np.inf
departure_eccentricity = 0.0

arrival_semi_major_axis = 3.8913e9
arrival_eccentricity = 0.999486


class Transfer:
    def __init__(self, fn, args=[], kwargs={}):
        self.obj = fn(*args, **kwargs)
        self.kwargs = kwargs
        self.args = args
        self.fn = fn

    def __repr__(self):
        return repr(self.obj)

    def __getattr__(self, name: str):
        return self.obj.__getattribute__(name)

    def __getstate__(self):
        out = self.__dict__.copy()
        if "obj" in out:
            del out["obj"]
        return out

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self.obj = self.fn(*self.args, **self.kwargs)

    def has_args(self, *args, **kwargs):
        if "args" in kwargs and "kwargs" in kwargs:
            kwargs = kwargs.pop("kwargs")
            args = kwargs.pop("args")

        return list(self.args) == list(args) and self.kwargs == kwargs


def fn(central_body, body_order, *args, departure_orbit, arrival_orbit, **kwargs):
    from tudatpy.numerical_simulation.environment_setup import create_simplified_system_of_bodies
    from tudatpy.trajectory_design.transfer_trajectory import (
        mga_settings_unpowered_unperturbed_legs,
        create_transfer_trajectory,
    )

    bodies = create_simplified_system_of_bodies()

    leg_settings, node_settings = mga_settings_unpowered_unperturbed_legs(
        body_order,
        departure_orbit=departure_orbit,
        arrival_orbit=arrival_orbit,
    )

    return create_transfer_trajectory(
        bodies,
        leg_settings,
        node_settings,
        body_order,
        central_body,
    )


t = Transfer(
    fn,
    args=[central_body, body_order],
    kwargs={
        "departure_orbit": (departure_semi_major_axis, departure_eccentricity),
        "arrival_orbit": (arrival_semi_major_axis, arrival_eccentricity),
    },
)

print(t)

d = pkl.dumps(t)
t = pkl.loads(d)
t


# %%
