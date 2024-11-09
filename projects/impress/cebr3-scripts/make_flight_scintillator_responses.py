"""
At long last, we decided on foils to put on IMPRESS, based on some Geant4 and analytical simulations:
    - 0um (C class detector)
    - 0um (low M class detector)
    - 100um (high M class detector)
    - 250um (X class detector)

The X-123 will have a 100um filter too.

This script just generates response files for the flight attenuators.
They are ONLY APPROXIMATE. Geant should be used for the real deal.
"""

import lzma
import os
import pickle

import astropy.units as u
import numpy as np

from adetsim.sim_src.FlareSpectrum import FlareSpectrum
from adetsim.hafx_src.HafxStack import HafxStack

attenuator_thicknesses = {
    "c1": 0 << u.um,
    "m1": 0 << u.um,
    "m5": 100 << u.um,
    "x1": 250 << u.um,
}


def main():
    out_dir = "responses"
    os.makedirs(out_dir, exist_ok=True)

    flare_energies = np.linspace(2, 400, num=1000) << u.keV
    fake_flux = np.zeros(flare_energies.size - 1)

    save = dict()

    for goes, t in attenuator_thicknesses.items():
        print("on", goes, t)
        fake_flare = FlareSpectrum(
            goes_class=goes,
            thermal=fake_flux,
            nonthermal=fake_flux,
            energy_edges=flare_energies.to_value(u.keV),
        )
        hs = HafxStack(enable_scintillator=True, att_thick=t.to_value(u.cm))

        # Do not want energy resolution.
        # This is just for "effective area" computations
        response = hs.generate_detector_response_to(fake_flare, disperse_energy=False)
        save[goes] = {
            "attenuator_thickness": t,
            "response_matrix": response << (u.ct / u.ph),
        }

    with lzma.open(f"{out_dir}/hafx_responses.pkl.xz", "wb") as f:
        pickle.dump(save, f)


if __name__ == "__main__":
    main()
