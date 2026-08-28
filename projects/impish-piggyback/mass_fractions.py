import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import astropy.units as u
import numpy as np
from astropy import constants
from astropy.table import QTable, Row
from skyfield.api import load, wgs84
from skyfield.jpllib import SpiceKernel
from skyfield.positionlib import Barycentric, Geocentric
from skyfield.timelib import Time as skyTime
from skyfield.toposlib import GeographicPosition
from skyfield.units import Distance
from skyfield.vectorlib import VectorSum

from adetsim.atmosphere import compute_lookup_table


@dataclass
class Position:
    lat: u.Quantity
    lon: u.Quantity
    alt: u.Quantity


@dataclass
class Observer:
    time: datetime
    lat: u.Quantity
    lon: u.Quantity
    alt: u.Quantity

    @property
    def utctime(self) -> datetime:
        """The observer time, in UTC."""
        return self.time.astimezone(timezone.utc)

    @property
    def skyfield_time(self) -> skyTime:
        """Returns the skyfield Time object."""
        return load.timescale().utc(
            year=self.utctime.year,
            month=self.utctime.month,
            day=self.utctime.day,
            hour=self.utctime.hour,
            minute=self.utctime.minute,
            second=self.utctime.second,
        )

    @property
    def wsg84(self) -> GeographicPosition:
        """The geographic position vector in the WGS84 frame."""
        return wgs84.latlon(
            latitude_degrees=self.lat.to_value(u.deg),
            longitude_degrees=self.lon.to_value(u.deg),
            elevation_m=cast(float, self.alt.to_value(u.m)),
        )

    @property
    def bcrs(self) -> Barycentric:
        """Barycentric BCRS position vector."""
        return cast(Barycentric, (self._earth + self.wsg84).at(self.skyfield_time))

    @property
    def icrs(self) -> Geocentric:
        """Geocentric ICRS position vector."""
        return cast(Geocentric, self.wsg84.at(self.skyfield_time))

    @property
    def to_sun(self) -> np.ndarray:
        """Unit vector pointing from the observer to the Sun."""
        vector_to_sun = cast(np.ndarray, self.bcrs.observe(self._sun).position.km)
        return vector_to_sun / np.linalg.norm(vector_to_sun)

    @property
    def _ephem(self) -> SpiceKernel:
        """Ephemeris file."""
        EPHEM_FILE: str = "de421.bsp"
        return cast(SpiceKernel, load(EPHEM_FILE))

    @property
    def _earth(self) -> VectorSum:
        """Earth ephemeris."""
        return self._ephem["earth"]

    @property
    def _sun(self) -> VectorSum:
        """Earth ephemeris."""
        return self._ephem["sun"]


def compute_atmosphere_slice_positions(
    observer: Observer, step: u.Quantity, stop: u.Quantity
) -> list[Position]:
    """Compute the latitude, longitude, and altitude at equally spaced steps
    between the observer and the Sun.
    """
    i = 0
    alt = observer.alt
    pos = cast(np.float64, observer.icrs.position.km)
    unit_vector = observer.to_sun
    steps: list[Position] = []
    while alt < stop:
        step_xyz = pos + i * unit_vector * step.to_value(u.km)
        step_pos = Geocentric(Distance(km=step_xyz).au, t=observer.skyfield_time)
        subpoint = wgs84.geographic_position_of(step_pos)
        steps.append(
            Position(
                subpoint.latitude.degrees << u.deg,
                subpoint.longitude.degrees << u.deg,
                subpoint.elevation.km << u.km,
            )
        )
        alt = subpoint.elevation.km << u.km
        i += 1
        print(
            f"Step {i:4d} | Lat: {subpoint.latitude.degrees:8.4f}° | "
            + f"Lon: {subpoint.longitude.degrees:8.4f}° | "
            + f"Alt: {subpoint.elevation.km:12.2f} km"
        )

    return steps


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
        "-l", type=float, default=-77.846, help="latitude, in degrees (+N/-S)"
    )
    _ = parser.add_argument(
        "-ll", type=float, default=166.668, help="longitude, in degrees (+E/-W)"
    )
    _ = parser.add_argument(
        "-o", type=float, default=40, help="observer altitude, in km"
    )
    _ = parser.add_argument(
        "-s", type=float, default=1, help="altitude slice step, in km"
    )
    _ = parser.add_argument(
        "-a", type=float, default=200, help="maximum altitude, in km"
    )
    _ = parser.add_argument(
        "-f",
        type=str,
        default=f"mass-fracs-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.asdf",
        help="output file name",
    )

    arg = parser.parse_args()
    outfile = arg.f
    impish = Observer(
        datetime.strptime(arg.t, "%Y-%m-%dT%H:%M:%S%z"),
        arg.l << u.deg,
        arg.ll << u.deg,
        arg.o << u.km,
    )
    step = arg.s << u.km
    stop = arg.a << u.km
    positions = compute_atmosphere_slice_positions(impish, step, stop)
    table = compute_mass_fractions_table(impish.utctime, positions)
    table.meta = {
        "observer datetime": impish.utctime,
        "observer lat": impish.lat,
        "observer lon": impish.lon,
        "line-of-sight step": step,
    }
    table.write(outfile)
    print(f"output table to {outfile}")


if __name__ == "__main__":
    main()
