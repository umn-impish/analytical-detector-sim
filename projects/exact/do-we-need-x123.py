import functools

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from adetsim.hafx_src.X123Stack import X123Stack
from adetsim.hafx_src.HafxStack import HafxStack
import adetsim.sim_src.FlareSpectrum as fs
from sunkit_spex.photon_power_law import compute_broken_power_law as broken_power


def main():
    sp = make_crab_xray_spectrum()
    res = make_responses(sp)
    rates = compute_rates(res, sp)
    print("rates per detector:")
    print(rates)

    num_scintillators = 4
    print(
        f"proportion: {100 * rates['x123'] / (num_scintillators * rates['hafx']):.2f}%"
    )


def progressor(func):
    # print progress of functions
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        print("start", func.__name__)
        ret = func(*args, **kwargs)
        print("end", func.__name__)
        return ret

    return wrap


@progressor
def compute_rates(rmats, spectrum):
    ret = dict()
    for name, dat in rmats.items():
        multiplied = dat["rmat"] @ spectrum.nonthermal
        total = np.sum(multiplied * np.diff(spectrum.energy_edges))
        total *= dat["area"]
        total = total << u.ph / u.s
        ret[name] = total
    return ret


@progressor
def make_responses(spec):
    xs = X123Stack(det_thick=(1 << u.mm).to(u.cm).value)
    hs = HafxStack(att_thick=0)

    return {
        "x123": {
            "area": xs.area,
            "rmat": xs.generate_detector_response_to(spec, disperse_energy=False),
        },
        "hafx": {
            "area": hs.area,
            "rmat": hs.generate_detector_response_to(spec, disperse_energy=False),
        },
    }


@progressor
def make_crab_xray_spectrum():
    # see https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node216.html#powerlaw
    NUSTAR_REF_ENERGY = 1 << u.keV
    # from Madsen+2017, doi:10.3847/1538-4357/aa6970
    power_law_index = 2.106
    normalization_at_1kev = 9.71

    num = 500
    a, b = 0.1, 200
    energies = np.logspace(np.log10(a), np.log10(b), num=num) << u.keV
    photon_flux = broken_power(
        energy_edges=energies,
        norm_energy=NUSTAR_REF_ENERGY,
        norm_flux=(normalization_at_1kev << u.ph / u.cm**2 / u.keV / u.s),
        break_energy=(1 << u.MeV),
        lower_index=power_law_index,
        upper_index=0,
    ).to(u.ph / u.cm**2 / u.keV / u.s)

    return fs.FlareSpectrum(
        goes_class="the crab",
        thermal=np.zeros_like(photon_flux.value),
        nonthermal=photon_flux.value,
        energy_edges=energies.value,
    )


if __name__ == "__main__":
    main()
