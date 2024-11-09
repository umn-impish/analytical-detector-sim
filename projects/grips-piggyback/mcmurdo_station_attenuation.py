import argparse
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os

from adetsim.atmoatt.atmospheric_attenuator import Atmosphere, generate_flare_spectrum
from adetsim.atmoatt.atmospheric_lookup_table import (
    plot_abundances,
    plot_abundances_stackplot,
    plot_densities,
)


def main():
    parser = argparse.ArgumentParser(
        description="Iterate a solar flare X-ray spectrum through the atmosphere. \
            at McMurdo Station, Antarctica (launch site of GRIPS-1).",
        epilog="Example of use: python mcmurdo-station-attenuation.py -f M5",
    )
    parser.add_argument("-f", type=str, help="flare GOES class, e.g. C1, M5, X8")
    parser.add_argument("-s", type=float, default=200, help="starting altitude, in km")
    parser.add_argument("-e", type=float, default=40, help="ending altitude, in km")

    arg = parser.parse_args()
    flare_class = arg.f
    maximum_altitude = arg.s * u.km
    minimum_altitude = arg.e * u.km
    zenith_angle = 54.8 * u.deg  # At solar noon, which is 01:56 PM

    atmo = Atmosphere(
        np.datetime64("2024-01-01T13:56"),
        -77.84 * u.degree,
        166.67 * u.degree,
        minimum_altitude,
        maximum_altitude,
        altitude_step=5 * u.km,
        solar_zenith=zenith_angle,
    )

    out_dir = f"./mcmurdo-station-attenuation/"
    os.makedirs(out_dir, exist_ok=True)

    plot_abundances(atmo.lookup_table)
    plt.savefig(f"{out_dir}/atmospheric_abundances.png")

    plot_abundances_stackplot(atmo.lookup_table)
    plt.savefig(f"{out_dir}/atmospheric_abundances_stacked.png")

    plot_densities(atmo.lookup_table)
    plt.savefig(f"{out_dir}/atmospheric_densities.png")

    atmo.attenuate_spectrum_through_layers(
        generate_flare_spectrum(flare_class), out_dir=out_dir
    )


if __name__ == "__main__":
    main()
