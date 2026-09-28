import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import astropy.units as u
import numpy as np
from astropy import constants
from astropy.table import QTable, Row

from adetsim.atmosphere import compute_lookup_table


@dataclass
class Position:
    lat: u.Quantity
    lon: u.Quantity
    alt: u.Quantity


def compute_mass_fractions(row: Row) -> dict[str, float]:
    """Computes mass fractions of each element from each species in the row."""
    # Molar masses from: https://www.webqc.org/mmcalc.php
    MOLAR_MASSES = {
        "N2": 28.01340 << u.g / u.mol,
        "O2": 31.99880 << u.g / u.mol,
        "O": 15.99940 << u.g / u.mol,
        "He": 4.0026020 << u.g / u.mol,
        "H": 1.007940 << u.g / u.mol,
        "Ar": 39.9480 << u.g / u.mol,
        "N": 14.00670 << u.g / u.mol,
        "Anomalous O": 15.99940 << u.g / u.mol,
        "NO": 30.00610 << u.g / u.mol,
    }
    # Get pyright to shut up
    c = lambda r: cast(u.Quantity, r)
    number_densities: dict[str, u.Quantity] = {
        "N": 2 * c(row["N2den"] + row["Nden"] + row["NOden"]),
        "O": 2 * c(row["O2den"] + row["Oden"] + row["Anomalous Oden"] + row["NOden"]),
        "He": c(row["Heden"]),
        "H": c(row["Hden"]),
        "Ar": c(row["Arden"]),
    }
    mass_fractions: dict[str, float] = {}
    for element, num_density in number_densities.items():
        mass_density: u.Quantity = num_density * MOLAR_MASSES[element] / constants.N_A  # pyright: ignore[reportAttributeAccessIssue]
        mass_fractions[element] = (mass_density / row["total mass density"]).value

    return mass_fractions


def compute_mass_fractions_table(
    utctime: datetime, positions: list[Position]
) -> QTable:
    """Compute the mass fractions at each atmospheric layer in positions."""
    rows: list[dict[str, float]] = []
    for position in positions:
        atmotable = compute_lookup_table(
            utctime,
            position.lat,
            position.lon,
            position.alt,
            remove_nonist_species=False,
        )
        row: Row = cast(Row, atmotable[0])
        slice_data: dict[str, float] = compute_mass_fractions(row)
        slice_data["altitude"] = position.alt << u.km
        slice_data["density"] = row["total mass density"] << u.g / (u.cm**3)
        rows.append(slice_data)

    return QTable(rows)


def main():

    parser = argparse.ArgumentParser(
        description="Compute mass fractions for each atmosphere slice along the line of sight \
            between the observer and the Sun.",
        epilog="Example of use: python mass-fractions.py -t 2026-01-01T13:56:00+1300",
    )
    _ = parser.add_argument(
        "-t",
        type=str,
        help="time with timezone relative to utc (%%Y-%%m-%%dT%%H:%%M:%%S%%z)",
        required=True,
    )
    _ = parser.add_argument(
        "-l", type=float, default=-77.846, help="observer latitude, in degrees (+N/-S)"
    )
    _ = parser.add_argument(
        "-ll",
        type=float,
        default=166.668,
        help="observer longitude, in degrees (+E/-W)",
    )
    _ = parser.add_argument(
        "-s", type=float, default=1, help="altitude slice step, in km"
    )
    _ = parser.add_argument(
        "-A", type=float, default=200, help="maximum altitude, in km"
    )
    _ = parser.add_argument(
        "-a", type=float, default=40, help="minimum altitude, in km"
    )
    _ = parser.add_argument(
        "-f",
        type=str,
        default=f"mass-fracs-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.asdf",
        help="output file name",
    )

    arg = parser.parse_args()
    outfile = arg.f
    lat = (arg.l << u.deg,)
    lon = (arg.ll << u.deg,)
    time = datetime.strptime(arg.t, "%Y-%m-%dT%H:%M:%S%z")
    start = arg.a
    step = arg.s
    stop = arg.A
    overhead = [
        Position(arg.l << u.deg, arg.ll << u.deg, a << u.km)
        for a in np.arange(start, stop, step)
    ]
    table = compute_mass_fractions_table(time, overhead)
    table.meta = {
        "observer datetime": time,
        "observer lat": lat,
        "observer lon": lon,
        "line-of-sight step": step << u.km,
    }
    table.write(outfile)
    print(f"output table to {outfile}")


if __name__ == "__main__":
    main()
