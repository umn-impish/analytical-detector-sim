import lzma
import os
import pickle
import scipy.stats as st

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sys

fig_dir = "figs"
os.makedirs(fig_dir, exist_ok=True)

plt.style.use(os.getenv("MPL_INTERACTIVE_STYLE"))

with open(sys.argv[1], "rb") as f:
    dat = pickle.load(f)

integration_time = 10 << u.s
x123_area = (17 << u.mm**2).to(u.cm**2).value
end_radius = np.sqrt(x123_area / np.pi)

edges = dat["edges"].to(u.keV).value
de = np.diff(edges)
# rads = np.arange(0.005, end_radius, 0.005)
# print(rads)
# for rad in rads:
rad = end_radius
fig, axs = plt.subplots(figsize=(14, 6), layout="constrained", nrows=2, ncols=2)
axs = axs.flatten()

ar = rad**2 * np.pi
for ax, (goes_class, sub_dat) in zip(axs, dat["dat"].items()):
    for k, spec in sub_dat.items():
        ct = spec * ar * de * integration_time.value
        draws = st.poisson.rvs(ct)
        ax.stairs(
            draws, edges, label=f"{k} ({ct.sum() / integration_time.value:.0f} cps)"
        )

    ax.set(
        xlabel="energy (keV)",
        ylabel="ct",
        title=f"{goes_class} GOES class simulated flare",
        xscale="log",
        yscale="log",
        ylim=(1, None),
    )
    ax.legend()

diam = 2 * rad
fig.suptitle(
    rf'counts in {integration_time} integration, aperture Ø = {diam*1e4:.1f}um, pinhole Ø = {dat["diameter"]:.2f}'
)
plt.show()
fig.savefig(f"{fig_dir}/{integration_time.value}s-{diam*1e4:.1f}um.pdf", dpi=300)
