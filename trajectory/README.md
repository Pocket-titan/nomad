# NOMAD trajectory

Trajectory generation with `transfer_trajectory`, then verification with `tudatpy` and/or `godot`.

## Todo
- [X] Wishlist generation
- [X] Figure out conda on DelftBlue
- [X] Multiprocessing/cpu on DelftBlue
- [ ] Figure out deployment on DelftBlue
- [X] Generate good boundary conditions etc.
- [ ] Add time constraint to `Problem`
- [ ] Keep track of champion in `main`; and log it
- [ ] Don't log exception info inside `fitness` function, just warn
- [X] Put `wishlist.pkl` inside `runs` folder; clean it etc.
- [ ] Calculate hyperbolic excess velocity as metric
- [ ] Add `batch_fitness` to `Problem`