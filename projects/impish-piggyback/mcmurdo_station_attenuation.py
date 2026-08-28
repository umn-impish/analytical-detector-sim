import argparse
import os
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from adetsim.atmosphere import (
    Atmosphere,
    generate_flare_spectrum,
    plot_abundances,
    plot_abundances_stackplot,
    plot_densities,
)
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Iterate a solar flare X-ray spectrum through the atmosphere. \
            at McMurdo Station, Antarctica (launch site of GRIPS-1).",
        epilog="Example of use: python mcmurdo-station-attenuation.py -f M5",
    )
    parser.add_argument(
        "-f", type=str, help="flare GOES class, e.g. C1, M5, X8", required=True
    )
    parser.add_argument(
        "-s", type=float, default=200, help="maximum altitude, in km", required=True
    )
    parser.add_argument(
        "-e", type=float, default=41, help="minimum altitude, in km", required=True
    )

    arg = parser.parse_args()
    flare_class = arg.f
    altitude_step = 5 * u.km
    altitudes = np.arange(arg.e, arg.s + altitude_step.value, altitude_step.value) * u.km
    zenith_angle = 54.8 << u.deg  # At solar noon, which is 01:56 PM

    atmo = Atmosphere(
        datetime.strptime("2024-01-01T13:56:00+1300", "%Y-%m-%dT%H:%M:%S%z"),
        -77.84 << u.degree,
        166.67 << u.degree,
        altitudes,
        solar_zenith=zenith_angle,
        photoelectric=True,
        rayleigh=True,
        compton=True,
    )

    out_dir = "./mcmurdo-station-attenuation/"
    os.makedirs(out_dir, exist_ok=True)

    plot_abundances(atmo.lookup_table)
    plt.savefig(f"{out_dir}/atmospheric_abundances.png")

    plot_abundances_stackplot(atmo.lookup_table)
    plt.savefig(f"{out_dir}/atmospheric_abundances_stacked.png")

    plot_densities(atmo.lookup_table)
    plt.savefig(f"{out_dir}/atmospheric_densities.png")


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        atmo.attenuate_spectrum_through_layers(
            generate_flare_spectrum(flare_class), out_dir=out_dir
        )


if __name__ == "__main__":
    main()
