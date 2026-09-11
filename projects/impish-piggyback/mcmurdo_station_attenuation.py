import argparse
import pathlib
import warnings
from datetime import datetime

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
        "-M",
        type=float,
        default=200,
        help="maximum (highest) altitude, in km",
        required=True,
    )
    parser.add_argument(
        "-m",
        type=float,
        default=41,
        help="minimum (lowest) altitude, in km",
        required=True,
    )
    parser.add_argument(
        "-p", action=argparse.BooleanOptionalAction, help="generate plots or not"
    )

    args = parser.parse_args()
    flare_class = args.f
    altitude_step = 1
    altitudes = np.arange(args.m, args.M + altitude_step, altitude_step) << u.km
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

    out_dir = pathlib.Path("./mcmurdo-station-attenuation/")
    out_dir.mkdir(exist_ok=True)

    if args.p:
        _ = plot_abundances(atmo.lookup_table)
        plt.savefig(out_dir / "atmospheric_abundances.png")
        plt.close("all")

        _ = plot_abundances_stackplot(atmo.lookup_table)
        plt.savefig(out_dir / "atmospheric_abundances_stacked.png")
        plt.close("all")

        _ = plot_densities(atmo.lookup_table)
        plt.savefig(out_dir / "atmospheric_densities.png")
        plt.close("all")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atmo.attenuate_spectrum_through_layers(
            generate_flare_spectrum(flare_class), out_dir=out_dir, plot_spectra=args.p
        )


if __name__ == "__main__":
    main()
