"""Call the appropriate MATLAB scripts and executable."""
from pathlib import Path
from subprocess import DEVNULL, STDOUT, Popen, PIPE
from os.path import join
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
    means: List[float] = [120, 120, 120],
    stds: List[float] = [10, 10, 10],
    fractions: List[float] = [0.33, 0.33],
):
    """Perform particle packing simulation.py

    Parameters
    ----------
    uid : int, optional
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
    eng = engine.start_matlab()
    eng.addpath(join("boppf", "utils"))

    means = double(list(means))
    stds = double(list(stds))
    fractions = np.append(fractions, 1 - np.sum(fractions))
    fractions = double(list(fractions))

    # generate input file
    Path(join("boppf", "data")).mkdir(exist_ok=True, parents=True)
    eng.write_input_file(uid, means, stds, fractions, particles, nargout=0)

    # run the particle packing simulation (executable)
    util_dir = join("boppf", "utils")
    data_dir = join("boppf", "data")
    fpath = join(util_dir, "run_executable.py")
    p = Popen(
        [executable, fpath], stdin=PIPE, stdout=DEVNULL, stderr=STDOUT, shell=True
    )
    idpath = join(data_dir, f"{uid}.inp")
    input = str.encode(idpath)
    p.communicate(input=input)

    # extract the volumetric packing fraction
    vol_frac = eng.read_vol_frac(uid, data_dir)
    print("vol_frac: ", vol_frac)

    eng.quit()

    return vol_frac
