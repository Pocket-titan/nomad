# NOMAD trajectory

Trajectory generation with `transfer_trajectory`, then verification with `tudatpy` and/or `godot`.

## Todo
- [X] Wishlist generation
- [X] Figure out conda on DelftBlue
- [X] Multiprocessing/cpu on DelftBlue
- [x] Figure out deployment on DelftBlue
- [X] Generate good boundary conditions etc.
- [~] Add time constraint to `Problem`
- [x] Keep track of champion in `main`; and log it
- [x] Don't log exception info inside `fitness` function, just warn
- [X] Put `wishlist.pkl` inside `runs` folder; clean it etc.
- [ ] Calculate hyperbolic excess velocity as metric
- [ ] Add `batch_fitness` to `Problem`

- [ ] Run cassini search validation
- [ ] Create Neptune low-fidelity search space
- [ ] Create low-thrust (hodographic?) function
- [ ] Create spherical? function
- [ ] Mixed legs?
- [ ] Run unpowered,dsm_velocity,hodographic searches