"""Perform a single particle packing simulation."""
from boppf.utils.particle_packing import particle_packing_simulation

dummy = True

if dummy:
    uid = "tmp"
    particles = 100
else:
    particles = 1500000

means = [120, 120, 120]
stds = [10, 10, 10]
fractions = [0.33, 0.33]

vol_frac = particle_packing_simulation(
    uid=uid, particles=particles, means=means, stds=stds, fractions=fractions
)
