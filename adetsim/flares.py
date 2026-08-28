import copy

import astropy.units as u
import numpy as np
from sunkit_spex.legacy import photon_power_law, thermal


def goes_class_lookup(flare_specifier: str) -> float:
    GOES_PREFIX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
    ch, mul = flare_specifier[0].upper(), float(flare_specifier[1:])
    return GOES_PREFIX[ch] * mul


class BattagliaParameters:
    # see Battaglia et al., https://arxiv.org/abs/astro-ph/0505154v1
    # equations: 11, 12, 8
    BELOW_INDEX = 1.5

    def __init__(self, goes_flux):
        pt_p1 = (np.log(goes_flux) / np.log(10)) << u.MK
        pt_p2 = (12 - np.log(3.5) / np.log(10)) << u.MK
        self.plasma_temp = (1 / 0.33) * (pt_p1 + pt_p2)

        self.emission_measure = (goes_flux / 3.6 * 1e50) ** (1 / 0.92) << (u.cm**-3)
        self.reference_energy = 35 << u.keV
        self.reference_flux = (goes_flux / (1.8e-5)) ** (1 / 0.83) << (
            u.ph / u.s / u.cm**2 / u.keV
        )

        if goes_flux < goes_class_lookup("C2"):
            self.spectral_index = (2.04 * self.reference_flux.value ** (-0.16)) << u.one
        else:
            self.spectral_index = (3.60 * self.reference_flux.value ** (-0.16)) << u.one

    def __repr__(self):
        return (
            "<"
            + ", ".join(
                [
                    f"Emission measure {self.emission_measure / 1e49:.3f}e49 particle2 / cm3",
                    f"Plasma temp {self.plasma_temp:.3f} MK",
                    f"Spectral index {self.spectral_index:.3f}",
                ]
            )
            + ">"
        )


class Flare:
    @u.quantity_input
    def __init__(self, goes_class: str, energy_edges: u.Quantity[u.keV]):
        bp = BattagliaParameters(goes_class_lookup(goes_class))
        self.goes_class: str = goes_class
        self.energy_edges: u.Quantity = copy.deepcopy(energy_edges)
        self.thermal: u.Quantity = thermal.thermal_emission(
            energy_edges=energy_edges,
            temperature=bp.plasma_temp,
            emission_measure=bp.emission_measure,
        )

        BATTAGLIA_BREAK_ENERGY = 5 << u.keV
        self.nonthermal: u.Quantity = photon_power_law.compute_broken_power_law(
            energy_edges,
            norm_energy=bp.reference_energy,
            norm_flux=bp.reference_flux,
            break_energy=BATTAGLIA_BREAK_ENERGY,
            lower_index=bp.BELOW_INDEX,
            upper_index=bp.spectral_index,
        )

        self.all_emission: u.Quantity = self.thermal + self.nonthermal  # pyright: ignore[reportAttributeAccessIssue]
