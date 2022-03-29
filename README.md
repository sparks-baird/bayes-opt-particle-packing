# bayes-opt-particle-packing

Bayesian optimization of particle packing fractions for nuclear fuels. The objective function (not released here) is based on proprietary code from Northrop Grumman Innovation Systems (NGIS).

To reproduce, this requires a proprietary executable (renamed to
`particle_packing_sim.exe`), a MATLAB file
for writing the input files (renamed to `write_input_file.m`), and a MATLAB file for reading the volume fraction from
the output files (renamed to `read_vol_frac.m`). These files should be placed into the [boppf/utils](boppf/utils).
