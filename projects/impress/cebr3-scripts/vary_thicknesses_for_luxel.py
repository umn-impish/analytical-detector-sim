"""
This script perturbs our "desired" attenuator thicknesses by some small amount (10%)
to see the effect on collected photons.

Not very useful now...
"""

import astropy.units as u
import numpy as np
import sys
import pickle

from adetsim.hafx_src.HafxStack import HafxStack
from adetsim.sim_src.FlareSpectrum import FlareSpectrum

MODEL_ENERGY_EDGES = np.arange(10, 300, 0.5)
DESIRED_THICKS = {
    # lower bound of 1 micron, upper bound of 20.
    # as long as we know the thickness with some precision (say +- 1%) then the exact value doesn't matter
    "m1": 1 << u.um,
    "m5": 225 << u.um,
    "x1": 530 << u.um,
}
NUM_PERTURBATIONS = 4
PERT_PROP = 0.1


def main():
    out = dict()
    key_fmt = "{}goes-{}"
    for cl, t in DESIRED_THICKS.items():
        fl = FlareSpectrum.make_with_battaglia_scaling(
            goes_class=cl, energy_edges=MODEL_ENERGY_EDGES
        )
        perturbs = np.linspace(
            (1 - PERT_PROP) * t, (1 + PERT_PROP) * t, NUM_PERTURBATIONS
        )
        for pt in perturbs:
            hs = HafxStack(enable_scintillator=True, att_thick=pt.to(u.cm).value)
            res = hs.generate_detector_response_to(fl, disperse_energy=True)
            att_flare = res @ fl.flare
            cr = (hs.area * att_flare * np.diff(MODEL_ENERGY_EDGES)).sum() << u.ct / u.s
            out[key_fmt.format(cl, pt)] = cr
            print(cl, pt, "->", cr)

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "luxel-thicks.pickle"
    with open(fn, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
