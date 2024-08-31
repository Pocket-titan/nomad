# %%
# champions = (
#     (
#         df.filter(pl.col("gen") > 3)
#         .group_by("gen")
#         .agg(pl.all().sort_by("dv").head(5))
#         .explode(pl.all().exclude("gen"))
#     )
#     .sort("dv")
#     .filter(pl.col("dv") < 20_000)
# )

# plt.scatter(champions["tof"], champions["dv"], c=champions["gen"], cmap="viridis_r")
# plt.xlabel("Time of flight [yr]")
# plt.ylabel(r"$\Delta$V [m/s]")

# # %%
# champions = (df.group_by("gen").agg(pl.all().sort_by("dv").first())).sort("gen")
# plt.plot(champions["gen"], champions["dv"])
# plt.xlabel("Generation")
# plt.ylabel(r"$\Delta$V [m/s]")
# plt.title("Best individual per generation")

# # %%
# selection = df.filter(pl.col("dv") <= 10_000)
# xs = np.array(selection.select(pl.col("x")).to_series().to_list())

# J2000 = Time("J2000", format="jyear_str")

# departures = J2000 + TimeDelta(xs[:, 0], format="sec")
# arrivals = J2000 + TimeDelta(np.sum(xs[:, : p.obj().number_of_legs + 1], axis=-1), format="sec")

# dvs = selection["dv"]

# plt.scatter(
#     departures.to_value("datetime64"),
#     arrivals.to_value("datetime64"),
#     cmap="viridis_r",
#     c=dvs,
# )
# plt.colorbar(label=r"$\Delta$V [m/s]")
# plt.xticks(rotation=-45)
# plt.xlabel("Departure date")
# plt.ylabel("Arrival date")


# # %%
# plt.plot(errs[0] + np.diff(errs))
