"""Call the appropriate MATLAB scripts and executable."""
from os import getcwd
import os
from pathlib import Path
from subprocess import DEVNULL, STDOUT, Popen, PIPE, run
from os.path import join, abspath
from sys import executable
from typing import List
import numpy as np

# conda activate boppf
# cd C:\Program Files\MATLAB\R2021a\extern\engines\python
# python setup.py install
from matlab import engine, double


def particle_packing_simulation(
    uid: str = "tmp",
    particles: int = int(1.5e6),
    means: List[float] = [120.0, 120.0, 120.0],
    stds: List[float] = [10.0, 10.0, 10.0],
    fractions: List[float] = [0.33, 0.33],
):
    """Perform particle packing simulation.py

    Parameters
    ----------
    uid : str, optional
        The prefix to the input filename, by default 0
    particles : int, optional
        The number of particles to drop in the simulation, by default 1500000
    means : List[float], optional
        The log-normal mean radius of the 3 particles, by default double([120, 120, 120])
    stds : List[float], optional
        The log-normal standard deviations of the 3 particles, by default double([10, 10, 10])
    fractions : List[float], optional
        The mass fractions of the first two particles, by default double([0.33, 0.33])

    Returns
    -------
    vol_frac : float
        Volumetric packing fraction of the lump of dropped particles.
    """
    cwd = os.getcwd()
    os.chdir(join("..", ".."))
    eng = engine.start_matlab()
    eng.addpath(join("boppf", "utils"))

    means = double(list(means))
    stds = double(list(stds))
    fractions = np.append(fractions, 1 - np.sum(fractions))
    fractions = double(list(fractions))

    # generate input file

    # run the particle packing simulation (executable)
    util_dir = join("boppf", "utils")
    data_dir = join("boppf", "data")

    Path(data_dir).mkdir(exist_ok=True, parents=True)
    eng.write_input_file(uid, means, stds, fractions, particles, data_dir, nargout=0)

    fpath = join(util_dir, "particle_packing_sim.exe")
    input = join(data_dir, f"{uid}.inp")
    run([fpath], input=input, text=True, stdout=PIPE, stderr=STDOUT)
    # with Popen([executable, fpath], stdin=PIPE, stdout=PIPE, stderr=STDOUT) as p:
    # out, err = p.communicate(input=input)
    # print(out, err)
    # print(getcwd())

    # extract the volumetric packing fraction
    vol_frac = eng.read_vol_frac(uid, data_dir)
    print("vol_frac: ", vol_frac)

    eng.quit()
    
    os.chdir(cwd)

    return vol_frac
