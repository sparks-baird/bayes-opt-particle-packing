"""Perform a single particle packing simulation."""
from os import chdir, getcwd, path

# import os
# import time

import numpy as np
import pandas as pd

# from psutil import cpu_count
import ray
from tqdm import tqdm
from boppf.utils.particle_packing import particle_packing_simulation
from uuid import uuid4
from timeit import default_timer as timer

import plotly.express as px

from boppf.utils.plotting import plot_and_save

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

dummy = False

skip_to_load = False

if dummy:
    uid = "tmp"
    particles = [1000, 2000, 3000]
    nreps = 5
else:
    particles = [
        int(0.3125e4),
        int(0.625e4),
        int(1.25e4),
        int(2.5e4),
        int(5.0e4),
        # int(7.5e4),
        # int(1.0e5),
    ]
    nreps = 6

means = [4.999999999, 1, 3.195027437]
stds = [0.541655268, 0.705662075, 0.755319764]
fractions = [0.840443899, 0.159556101]

seeds = np.array(list(range(10, 10 + len(particles) + 1)))

savepath = path.join("figures", "runtime")

if not skip_to_load:

    cwd = getcwd()
    chdir("boppf/utils")
    avg_dts = []
    std_dts = []
    avg_vol_fracs = []
    std_vol_fracs = []
    for particle in tqdm(particles):
        dts = []
        vol_fracs = []

        @ray.remote
        def get_time_vol_frac(seed):
            uid2 = str(uuid4())[0:8]
            start = timer()
            vol_frac = particle_packing_simulation(
                uid=uid2,
                particles=particle,
                means=means,
                stds=stds,
                fractions=fractions,
                verbose=False,
                seed=seed,
            )
            dt = timer() - start
            print(f"elapsed time (s): {dt:.2f}")
            return dt, vol_frac

        results = ray.get([get_time_vol_frac.remote(seed) for seed in seeds])
        seeds = np.array(seeds) + len(particles)  # to ensure no repeat seeds
        dts, vol_fracs = np.array(results).T

        avg_dt = np.mean(dts)
        std_dt = np.std(dts)
        avg_vol_frac = np.mean(vol_fracs)
        std_vol_frac = np.std(vol_fracs)

        avg_dts.append(avg_dt)
        std_dts.append(std_dt)
        avg_vol_fracs.append(avg_vol_frac)
        std_vol_fracs.append(std_vol_frac)
    chdir(cwd)

    df = pd.DataFrame(
        {
            "particles": particles,
            "runtime": avg_dts,
            "t_std": std_dts,
            "vol_frac": avg_vol_fracs,
            "v_std": std_vol_fracs,
        }
    )
    df.to_csv(savepath + ".csv", index=False)
else:
    df = pd.read_csv(savepath + ".csv")
    particles = df["particles"]
    avg_vol_fracs = df["vol_frac"]
    std_vol_fracs = df["v_std"]
    avg_dts = df["runtime"]
    std_dts = df["t_std"]

fig = px.scatter(
    df,
    x="particles",
    y="vol_frac",
    error_y="v_std",
    labels={"particles": "number of particles", "vol_frac": "volume fraction"},
)
plot_and_save(savepath + "_vol_frac", fig, show=True, update_legend=True)

fig = px.scatter(
    df,
    x="particles",
    y="runtime",
    error_y="t_std",
    labels={"particles": "number of particles", "runtime": "runtime (s)"},
)
plot_and_save(savepath + "_runtime", fig, show=True, update_legend=True)

1 + 1

# %% Code Graveyard
# long_df = df.rename(columns={"runtime": "runtime (s)"}).melt(
#     id_vars="particles",
#     value_vars=["vol_frac", "runtime (s)"],
#     value_name="y",
#     var_name="y_name",
# )
# long_df["std"] = df[["v_std", "t_std"]].unstack().values
# fig = px.scatter(
#     long_df,
#     x="particles",
#     y="y",
#     error_y="std",
#     color="y_name",
#     labels={"particles": "number of particles"},
# )
# fig.update_layout(legend_title="")

# fig = make_subplots(specs=[[{"secondary_y": True}]])
# fig.add_trace(
#     go.Scatter(
#         x=particles,
#         y=avg_vol_fracs,
#         error_y=dict(type="data", array=std_vol_fracs, visible=True),
#         name="vol_frac",
#     ),
#     secondary_y=False,
# )
# fig.add_trace(
#     go.Scatter(
#         x=particles,
#         y=avg_dts,
#         error_y=dict(type="data", array=std_dts, visible=True),
#         name="runtime (s)",
#     ),
#     secondary_y=True,
# )
# fig.update_layout(title_text="")
# fig.update_xaxes(title_text="number of particles")
# fig.update_yaxes(title_text="volume fraction", secondary_y=False)
# fig.update_yaxes(title_text="runtime (s)", secondary_y=True)

# plot_and_save(savepath, fig, show=True, update_legend=True)
