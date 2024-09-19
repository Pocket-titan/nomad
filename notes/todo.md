# Optimization notes

Really "zeros in" on optimum solution (not just local?)
```python
t=pg.fully_connected(),
s_pol=pg.s_policies.select_best(0.1),
r_pol=pg.r_policies.fair_replace(0.1),
```

- sade > mbh_sade

- pso does what mbh does but better!
- pso, sade: both unconnected
- sga is trash, so is mbh_sga
- sade > gaco > de1220 > pso
- mbh_gaco is kinda good though... (with higher generations (50))

- try expanding bounds for cassini to see if it still finds the same optima
- run both pso and sade on neptune combs, yearly or all at once

gaco, mbh_gaco, sade(, pso)


### Unpowered
total:
- 4 bounds; in 3 year bins
- 4 algos: gaco, mbh_gaco, sade, pso
- 3 orders:
    - ["Earth", "Jupiter", "Neptune"]
    - ["Earth", "Mars", "Jupiter", "Neptune"]
    - ["Earth", "Venus", "Earth", "Earth", "Jupiter", "Neptune"]
- single objective: min delta-v
- departure, arrival at np.inf, e=0
- evolve_kwargs:
  {
    "num_evolutions": 50,
    "num_generations": 25,
    "pop_size": 50,
  }

### Powered (1dsm)
- 2 orders:
  - ["Earth", "Neptune"]
  - ["Earth", "Jupiter", "Neptune"]
      - dsm during EJ leg, and JN leg (index 0 or 1)
- 4 algos: gaco, mbh_gaco, sade, pso
- single objective: min delta-v
- departure, arrival at np.inf, e=0
- evolve_kwargs:
  {
    "num_evolutions": 50,
    "num_generations": 25,
    "pop_size": 50,
  }

## All
-> if time:
- multi-objective around best solutions
- porkchop plots around best solutions
- high-fidelity sim (after writing results)