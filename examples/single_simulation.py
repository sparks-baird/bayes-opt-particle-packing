"""Perform a single particle packing simulation."""
from os import chdir, getcwd
from boppf.utils.particle_packing import particle_packing_simulation
from uuid import uuid4

dummy = True

if dummy:
    uid = "tmp"
    particles = 10
else:
    uid = str(uuid4())[0:8]
    particles = int(2.5e5)

means = [120.0, 120.0, 120.0]
stds = [10.0, 10.0, 10.0]
fractions = [0.33, 0.33]

# means = [26.6259030268466, 343.511344840641, 10.0]
# stds = [173.089078783001, 606.693903338546, 606.693903338546]
# fractions = [0.252296349186048, 0.747703650813951, 0.0]

cwd = getcwd()
chdir("boppf/utils")
vol_frac = particle_packing_simulation(
    uid=uid, particles=particles, means=means, stds=stds, fractions=fractions
)
chdir(cwd)

1 + 1
