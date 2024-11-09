from typing import Tuple
import astropy.units as u
import functools
import numpy as np

try:
    import sunkit_spex.thermal as thermal
except ModuleNotFoundError:
    import sunkit_spex.legacy.thermal as thermal

GOES_PREFIX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

BATTAGLIA_BREAK_ENERGY = 5


def goes_class_lookup(flare_specifier: str) -> np.float64:
    ch, mul = flare_specifier[0].upper(), float(flare_specifier[1:])
    return GOES_PREFIX[ch] * mul


def power_law_integral(energy=0, norm_energy=1, norm=1, index=1):
    """assumes f(E) = norm * (energy / norm_energy)**(-index)
    (index > 0)
    works for index < 0 but keep functional form in mind (pass in -index)
    """
    prefactor = norm * norm_energy
    arg = energy / norm_energy
    if index == 1:
        return prefactor * np.log(arg)
    return prefactor * arg ** (1 - index) / (1 - index)


def broken_power_law_binned_flux(
    energy_edges,
    reference_energy,
    reference_flux,
    break_energy,
    lower_index,
    upper_index,
):
    """use analytically integrated broken power law to get flux in given energy bins"""
    norm_idx = lower_index if reference_energy <= break_energy else upper_index
    # norm is from continuity at break energy and solving.
    norm = reference_flux * (reference_energy / break_energy) ** (norm_idx)

    cnd = energy_edges <= break_energy
    lower = energy_edges[cnd]
    upper = energy_edges[~cnd]

    up_integ = functools.partial(
        power_law_integral, norm_energy=break_energy, norm=norm, index=upper_index
    )
    low_integ = functools.partial(
        power_law_integral, norm_energy=break_energy, norm=norm, index=lower_index
    )

    lower_portion = low_integ(energy=lower[1:])
    lower_portion -= low_integ(energy=lower[:-1])

    upper_portion = up_integ(energy=upper[1:])
    upper_portion -= up_integ(energy=upper[:-1])

    twixt_portion = []
    # bin between the portions is comprised of both power laws
    if lower.size > 0 and upper.size > 0:
        twixt_portion = np.diff(low_integ(energy=np.array([break_energy, upper[0]])))
        twixt_portion += np.diff(up_integ(energy=np.array([lower[-1], break_energy])))

    ret = np.concatenate((lower_portion, twixt_portion, upper_portion))
    assert ret.size == (energy_edges.size - 1)
    # go back to units of cm2/sec/keV
    return ret / np.diff(energy_edges)


class BattagliaParameters:
    # see Battaglia et al., https://arxiv.org/abs/astro-ph/0505154v1
    # equations: 11, 12, 8
    BELOW_INDEX = 1.5

    def __init__(self, goes_flux):
        pt_p1 = np.log(goes_flux) / np.log(10)  # MK
        pt_p2 = 12 - np.log(3.5) / np.log(10)  # MK
        self.plasma_temp = (1 / 0.33) * (pt_p1 + pt_p2)  # MK

        self.emission_measure = (goes_flux / 3.6 * 1e50) ** (
            1 / 0.92
        )  # particle^2 / cm3
        self.reference_energy = 35.0  # keV
        self.reference_flux = (goes_flux / (1.8e-5)) ** (
            1 / 0.83
        )  # photon / (s cm2 keV)

        if goes_flux < goes_class_lookup("C2"):
            self.spectral_index = 2.04 * self.reference_flux ** (-0.16)  # unitless
        else:
            self.spectral_index = 3.60 * self.reference_flux ** (-0.16)  # unitless

    def gen_vth_params(self) -> Tuple[float, float]:
        K_B = 8.627e-8  # keV/K
        """
        generate plasma temperature, emission measure in units appropriate for idl function f_vth.
        returns: (plasma_temp, emission_measure) in units of (keV, 1e49 * particles2 / cm3)
        """
        pt = self.plasma_temp * K_B * 1e6
        em = self.emission_measure / 1e49
        return {"pt": pt, "em": em}

    def __repr__(self):
        return (
            "<"
            + ", ".join(
                [
                    f"Emission measure {self.emission_measure/1e49:.3f}e49 particle2 / cm3",
                    f"Plasma temp {self.plasma_temp:.3f} MK",
                    f"Spectral index {self.spectral_index:.3f}",
                ]
            )
            + ">"
        )


class FlareSpectrum:
    _ENERGY_LIMITS = (1.001, 200.15)

    @classmethod
    def make_with_battaglia_scaling(
        cls,
        goes_class: str,
        energy_edges: np.ndarray,
        rel_abun=1.0,
        break_energy=BATTAGLIA_BREAK_ENERGY,
    ):
        """goes flux in W/m2, energies in keV"""
        bp = BattagliaParameters(goes_class_lookup(goes_class))

        edges = energy_edges

        over_cnd = edges > FlareSpectrum._ENERGY_LIMITS[0]
        under_cnd = edges < FlareSpectrum._ENERGY_LIMITS[1]
        thermal_edges = edges[over_cnd & under_cnd]

        thermal_spec = thermal.thermal_emission(
            thermal_edges * u.keV,
            bp.plasma_temp * u.MK,
            bp.emission_measure * (u.cm ** (-3)),
        ).value

        # make thermal/nonthermal spectra the same size
        verify = lambda n: n if n > 0 else 0
        under_elts, over_elts = np.sum(~over_cnd), np.sum(~under_cnd)
        under_pad, over_pad = np.zeros(verify(under_elts)), np.zeros(verify(over_elts))
        thermal_spec = np.concatenate((under_pad, thermal_spec, over_pad))

        # change to binned version
        nonthermal_spec = broken_power_law_binned_flux(
            energy_edges=edges,
            reference_energy=bp.reference_energy,
            reference_flux=bp.reference_flux,
            break_energy=break_energy,
            lower_index=bp.BELOW_INDEX,
            upper_index=bp.spectral_index,
        )

        return cls(
            goes_class=goes_class,
            energy_edges=energy_edges,
            thermal=thermal_spec,
            nonthermal=nonthermal_spec,
        )

    def __init__(
        self,
        goes_class: str,
        thermal: np.ndarray,
        nonthermal: np.ndarray,
        energy_edges: np.ndarray,
    ):
        self.goes_class = goes_class
        self.energy_edges = energy_edges
        self.thermal, self.nonthermal = thermal, nonthermal

    @property
    def goes_flux(self) -> np.float64:
        return goes_class_lookup(self.goes_class)

    @property
    def flare(self) -> np.ndarray:
        try:
            return self.thermal + self.nonthermal
        except ValueError:
            print(f"thermal: {self.thermal.size}\tnonthermal: {self.nonthermal.size}")
            raise

    def __repr__(self):
        ret = "[FlareSpectrum: "
        ret += f"energy_edges=(start={self.energy_edges[0]:.2f}, stop={self.energy_edges[-1]:.2f}), "
        ret += f"goes_class={self.goes_class}"

        return ret + "]"
