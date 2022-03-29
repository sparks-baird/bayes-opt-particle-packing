"""Perform a single particle packing simulation."""
from boppf.utils.particle_packing import particle_packing_simulation

dummy = True

if dummy:
    id = "tmp"
    particles = 100
else:
    particles = 1500000

particle_packing_simulation(id, particles)
