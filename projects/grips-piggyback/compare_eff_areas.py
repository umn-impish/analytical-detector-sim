import copy
import os
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import grips_stack as gs
from adetsim.sim_src import FlareSpectrum as fs

plt.style.use(os.getenv("MPL_INTERACTIVE_STYLE"))

geom_area = ((40 * 40) << u.mm**2).to(u.cm**2)


def main():
    PLA_INFILL = 10 / 100  # percent
    PLA_THICKNESS = PLA_INFILL * (1 << u.mm)
    thicks = {
        "Al": (50 << u.um).to(u.cm).value,
        "pla": (PLA_THICKNESS << u.mm).to(u.cm).value,
        "teflon": (0.1 << u.mm).to(u.cm).value,
        "gagg": (5 << u.mm).to(u.cm).value,
    }

    edges = np.logspace(0, np.log10(300), num=1000)
    spec = np.zeros(edges.size - 1)
    dummy = fs.FlareSpectrum("n/a", spec, spec, edges)
    srm_fn = "lyso-gagg-srms.pkl"
    if not os.path.exists(srm_fn):
        gagg_stack = gs.GripsStack(thicks)

        thicks = copy.deepcopy(thicks)
        thicks.pop("gagg")
        thicks["lyso"] = (5 << u.mm).to_value(u.cm)
        lyso_stack = gs.GripsStack(thicks)

        srms = dict()
        srms["gagg"] = gagg_stack.generate_detector_response_to(
            dummy, disperse_energy=False
        )
        srms["lyso"] = lyso_stack.generate_detector_response_to(
            dummy, disperse_energy=False
        )
        with open(srm_fn, "wb") as f:
            pickle.dump(srms, f)
    else:
        with open(srm_fn, "rb") as f:
            srms = pickle.load(f)

    color_map = {"gagg": "black", "lyso": "red"}
    name_map = {"gagg": "GAGG", "lyso": "LYSO"}

    fig, ax = plt.subplots(layout="constrained")
    area_vector = geom_area * np.ones(edges.size - 1)
    for name, srm in srms.items():
        eff_area = srm @ area_vector
        ax.stairs(eff_area, edges, color=color_map[name], label=name_map[name])
    ax.set(
        xlabel="Energy (keV)",
        ylabel="Area (cm$^2$)",
        title="Effective area curves: LYSO & GAGG ALXS-HXR",
        xscale="log",
        yscale="log",
        xlim=(3, 300),
        ylim=(0.1, 20),
    )
    ax.legend()
    fig.savefig("gagg-vs-lyso.pdf", dpi=400)


if __name__ == "__main__":
    main()
