"""Perform a single particle packing simulation."""
from os import chdir, getcwd, path

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from boppf.utils.particle_packing import particle_packing_simulation
from uuid import uuid4
from timeit import default_timer as timer

import plotly.express as px

from boppf.utils.plotting import plot_and_save

dummy = False

if dummy:
    uid = "tmp"
    particles = [100, 200]
    nreps = 10
else:
    particles = [
        int(0.3125e4),
        int(0.625e4),
        int(1.25e4),
        int(2.5e4),
        int(5.0e4),
        int(7.5e4),
        int(1.0e5),
    ]
    nreps = 10

means = [4.999999999, 1, 3.195027437]
stds = [0.541655268, 0.705662075, 0.755319764]
fractions = [0.840443899, 0.159556101]


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
    def get_time_vol_frac():
        if not dummy:
            uid2 = str(uuid4())[0:8]
        else:
            uid2 = uid
        start = timer()
        vol_frac = particle_packing_simulation(
            uid=uid2,
            particles=particle,
            means=means,
            stds=stds,
            fractions=fractions,
            verbose=False,
        )
        dt = timer() - start
        print(f"elapsed time (s): {dt:.2f}")
        return dt, vol_frac

    results = ray.get([get_time_vol_frac.remote() for _ in range(nreps)])
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
savepath = path.join("figures", "runtime")
df.to_csv(savepath + ".csv")

long_df = df.rename(columns={"runtime": "runtime (s)"}).melt(
    id_vars="particles",
    value_vars=["vol_frac", "runtime (s)"],
    value_name="y",
    var_name="y_name",
)
long_df["std"] = df[["v_std", "t_std"]].unstack().values
fig = px.scatter(
    long_df,
    x="particles",
    y="y",
    error_y="std",
    color="y_name",
    labels={"particles": "number of particles"},
)
fig.update_layout(legend_title="")
plot_and_save(savepath, fig, show=True, update_legend=True)

1 + 1
