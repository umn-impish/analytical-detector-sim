import numpy as np
from adetsim.sim_src.FlareSpectrum import FlareSpectrum

"""
Compute the count rate of a Battaglia model flare given a start and end energy.
"""

goes_class = "X1"
start_energy = 1
end_energy = 100
de = 0.1
edges = np.arange(start_energy, end_energy + de, step=de)

fs = FlareSpectrum.make_with_battaglia_scaling(
    goes_class=goes_class, energy_edges=edges
)

eng_cut = (fs.energy_edges >= start_energy) & (fs.energy_edges <= end_energy)

e, f = fs.energy_edges, fs.flare[eng_cut[:-1]]

countrate = np.sum(f * np.diff(e)[eng_cut[:-1]])
print(
    f"count rate for {goes_class} flare from {start_energy} to {end_energy} is: {countrate:.3e} phot/sec/cm2"
)
