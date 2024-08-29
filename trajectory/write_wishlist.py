# %%
import cloudpickle as pkl
import numpy as np


defaults = dict(
    body_order=None,
    create_obj=None,
    p_kwargs=dict(
        bounds=dict(),
        fixed=dict(),
        dim=1,
    ),
    evolve_kwargs=dict(
        num_evolutions=50,
        num_generations=25,
        pop_size=10,
        seed=4444,
        algo=None,
    ),
    suffix="",
)


def main():
    # if file exists, load it
    try:
        with open("wishlist.pkl", "rb") as f:
            prev = pkl.load(f)
    except FileNotFoundError:
        prev = []

    wishlist = [*prev, defaults]

    with open("wishlist.pkl", "wb") as f:
        pkl.dump(wishlist, f)


if __name__ == "__main__":
    main()

# %%
pkl.load(open("wishlist.pkl", "rb"))
