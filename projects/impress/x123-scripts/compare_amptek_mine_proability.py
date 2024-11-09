import matplotlib.pyplot as plt
import numpy as np

import adetsim.sim_src.FlareSpectrum as FS
from adetsim.hafx_src.X123CdTeStack import X123CdTeStack

AMPTEK_FN = "amptek-data-x123-cdte.tab"

edges = np.arange(1, 500.1, 0.1)
fs = FS.FlareSpectrum.make_with_battaglia_scaling(goes_class="C1", energy_edges=edges)
print("done flare spectrum")

xs = X123CdTeStack()
print("make response matrix")
mtx = xs.generate_detector_response_to(fs, disperse_energy=False)
print("done response matrix")
my_engs = fs.energy_edges
my_prob = mtx @ np.ones_like(fs.flare)
print("done multiply prob mine")

am_engs, am_prob = np.loadtxt(AMPTEK_FN, unpack=True)

fig, ax = plt.subplots()

ax.plot(am_engs, am_prob, label="Amptek provided")
ax.stairs(my_prob * 100, my_engs, label="My calculated")

ax.legend()
ax.set(
    title="Compare model and company probabilities, X-123 CdTe",
    ylabel="Interaction probability (%)",
    xlabel="Energy (keV)",
    xscale="log",
    yscale="log",
)

fig.tight_layout()
fig.set_size_inches(8, 6)
plt.show()
