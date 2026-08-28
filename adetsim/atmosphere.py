import copy
import itertools
import os
import pickle
from collections.abc import Mapping, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import astropy.units as u
import matplotlib.axes as mpa
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable, Row
from astropy.visualization import quantity_support
from pymsis import msis

from . import flares, matter

style_file = Path(__file__).parent / "styles/plot.mplstyle"

EARTH_RADIUS = 6378 * u.km

MARKERS = itertools.cycle(("x", "+", "o", "*", "v", "s", "h", "D"))
ACCEPTED_ELEMENTS = [
    "N2",
    "O2",
    "O",
    "He",
    "H",
    "Ar",
    "N",
]  # Excludes Anomalous O and NO since they're not indexed by NIST
DENSITY_COLS = [f"{e}den" for e in ACCEPTED_ELEMENTS]
ABUNDANCE_COLS = [f"{e}abund" for e in ACCEPTED_ELEMENTS]


def thickness_through_zenith(
    zenith_angle: u.Quantity,
    atmospheric_radius: u.Quantity = 200 * u.km,
    observer_altitude: u.Quantity = 0 * u.km,
) -> u.Quantity:
    """
    The atmospheric radius should correspond to the thickness as seen by the X-rays,
    i.e. the maximum altitude used in the simulation.
    """

    zenith_angle = zenith_angle << u.radian

    sqrt = np.sqrt(
        (EARTH_RADIUS + observer_altitude) ** 2 * (np.cos(zenith_angle)) ** 2
        + atmospheric_radius**2
        - observer_altitude**2
        + 2 * EARTH_RADIUS * (atmospheric_radius - observer_altitude)
    )

    return sqrt - (EARTH_RADIUS + observer_altitude) * np.cos(zenith_angle)


def _convert_msis_output_to_qtable(run_output: np.ndarray) -> QTable:
    """
    Converts the output from a MSIS run to an Astropy QTable.
    """

    columns = {
        "total mass density": u.kg / (u.m**3),
        "N2den": u.m ** (-3),
        "O2den": u.m ** (-3),
        "Oden": u.m ** (-3),
        "Heden": u.m ** (-3),
        "Hden": u.m ** (-3),
        "Arden": u.m ** (-3),
        "Nden": u.m ** (-3),
        "Anomalous Oden": u.m ** (-3),
        "NOden": u.m ** (-3),
        "Temperature": u.Kelvin,
    }

    table = QTable(run_output, names=columns.keys(), units=columns.values())
    table["total mass density"] = table["total mass density"] << u.g / (u.cm**3)
    for c in table.columns:
        if c[-3:] == "den":
            table[c] = table[c] << u.cm ** (-3)

    return table


def _compute_abundances(data: QTable):
    """
    Computes atmospheric abundances from the densities of each element and molecule.
    Adds columns to the provided table.
    """

    indices = [list(data.columns).index(c) + 1 for c in DENSITY_COLS]
    abund_cols = [c.replace("den", "") + "abund" for c in DENSITY_COLS]

    abundances = np.array([cast(u.Quantity, data[c]).value for c in DENSITY_COLS])
    totals = np.sum(abundances, axis=0)
    abundances = abundances / totals
    data.add_columns(list(abundances), indexes=indices, names=abund_cols)


def compute_lookup_table(
    time: datetime,
    lat: u.Quantity,
    lon: u.Quantity,
    altitudes: Iterable[u.Quantity],
    remove_nonist_species: bool = True,
    **run_kwargs,
) -> QTable:
    """
    Computes a lookup table for atmospheric composition using MSIS.
    time should either be in UTC or be timezone-aware.
    """

    lat = lat << u.degree
    lon = lon << u.degree
    output = msis.run(
        dates=time.astimezone(timezone.utc),
        lons=lon.value,
        lats=lat.value,
        alts=altitudes.to_value(u.km),
        **run_kwargs,
    )
    output = np.squeeze(output)
    output[np.isnan(output)] = 0
    table = _convert_msis_output_to_qtable(output)
    table["altitude"] = altitudes
    table.meta = {"datetime": datetime, "lat": lat, "lon": lon}

    if remove_nonist_species:
        for c in (table.columns).copy():
            if c[-3:] == "den" and c not in DENSITY_COLS:
                table.remove_column(c)
                print("removed", c)

    _compute_abundances(table)

    return table


def plot_densities(data: QTable) -> mpa.Axes:
    _, ax = plt.subplots(figsize=(8, 6), layout="constrained", sharex=True)

    altitude = cast(u.Quantity, data["altitude"])
    marker = itertools.cycle(("x", "+", "o", "*", "v", "s"))
    for column in DENSITY_COLS:
        if "den" in column and isinstance(data[column], u.Quantity):
            label = column.replace("den", "")
            ax.plot(
                altitude,
                cast(u.Quantity, data[column]),
                marker=next(marker),
                ls="--",
                markersize=6,
                lw=0.25,
                label=label,
            )
    ax.plot(
        altitude,
        cast(u.Quantity, data["total mass density"]),
        marker=next(marker),
        ls="--",
        markersize=3,
        lw=0.25,
        label="air",
    )
    ax.axvline(40, c="gray", ls=":")

    ax.set(
        xlabel=f"Altitude ({altitude.unit})",
        ylabel=f"Density ({cast(u.Quantity, data[column]).unit})",
        yscale="log",
        ylim=(1e-20, ax.get_ylim()[1]),
    )
    ax.legend()

    return ax


def plot_abundances(data: QTable, ax: mpa.Axes | None = None) -> mpa.Axes:
    with plt.style.context(style_file):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6), layout="constrained", sharex=True)

        altitude = cast(u.Quantity, data["altitude"])

        for column in data.columns:
            if "abund" in column:
                label = column.replace("abund", "")
                ax.plot(
                    altitude,
                    cast(np.ndarray, data[column]),
                    marker=next(MARKERS),
                    ls="--",
                    markersize=6,
                    lw=0.3,
                    label=label,
                )
        ax.axvline(40, c="gray", ls=":")

        meta: Mapping[str, object] = cast(dict, data.meta)
        ax.set(
            xlabel=f"Altitude ({altitude.unit})",
            ylabel="Abundance",
            title=f"Atmospheric Abundances on {meta['datetime']}, {meta['lat']}, {meta['lon']}",
            yscale="log",
            xlim=(altitude.min().value, altitude.max().value),
            ylim=(1e-5, 1),
        )
        ax.legend()

    return ax


def plot_abundances_stackplot(data: QTable, ax: mpa.Axes | None = None) -> mpa.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    abund_cols = [c.replace("den", "") + "abund" for c in DENSITY_COLS]
    labels = [c.replace("abund", "") for c in abund_cols]
    y = np.array([list(r.values()) for r in data[abund_cols]]).T * 100
    altitude = cast(u.Quantity, data["altitude"])

    ax.stackplot(altitude.value, y, labels=labels)

    meta: Mapping[str, object] = cast(dict, data.meta)
    ax.set(
        xlabel=f"Altitude ({altitude.unit})",
        ylabel="Abundance",
        title=f"Atmospheric Abundances on {meta['datetime']}, {meta['lat']}, {meta['lon']}",
        yscale="log",
        xlim=(altitude.min().value, altitude.max().value),
        ylim=(75, 100),
    )
    ax.legend()

    return ax


def generate_flare_spectrum(
    goes_class: str,
    min_energy: float = 2,
    max_energy: float = 300,
    energy_delta: float = 0.1,
) -> flares.Flare:
    """
    Energy parameters are in keV.
    energy_delta is the width of each energy bin.
    """

    edges = np.arange(min_energy, max_energy + energy_delta, energy_delta)
    return flares.Flare(goes_class=goes_class, energy_edges=edges << u.keV)


@dataclass
class Atmosphere:
    time: datetime
    latitude: u.Quantity[u.deg]
    longitude: u.Quantity[u.deg]
    altitudes: u.Quantity[u.km]
    cross_section_diameter: u.Quantity[u.cm] = 10 * u.cm
    solar_zenith: u.Quantity[u.deg] = 0 * u.deg

    # Flags for what attenuation we want enabled/disabled for
    # atmospheric layers
    photoelectric: bool = True
    rayleigh: bool = False
    compton: bool = False

    def __post_init__(self):
        self.lookup_table = compute_lookup_table(
            self.time,
            self.latitude,
            self.longitude,
            self.altitudes,
        )

    def _get_altitude_table_row(self, altitude: u.Quantity) -> Row:
        """
        Helper function to retrieve data for a specific altitude.
        If the specified altitude is not in the table, this function
        returns the data for the altitude closest to that specified.
        """
        diffs = np.abs(self.lookup_table["altitude"] - altitude)
        index = np.where(diffs == diffs.min())[0][0]
        return cast(Row, self.lookup_table[index])

    def _compute_layer_composition(self, row: Row) -> dict[str, float]:
        """
        Computes the **elemental** abundances for the provided row.

        If there is a corresponding molecular form for the element
        (e.g. oxygen), then the composition accounts for the two
        oxygen atoms per molcule when computing the new abundances.
        """

        abundances = cast(Row, row[ABUNDANCE_COLS])
        composition = {}
        for element in abundances.colnames:
            element_name = element.replace("abund", "")
            if element_name[-1].isnumeric():
                multiplier = int(element_name[-1])
                element_name = element_name[:-1]
            else:
                multiplier = 1

            if element_name in composition:
                composition[element_name] += multiplier * row[element]
            else:
                composition[element_name] = multiplier * row[element]

        total = np.sum(list(composition.values()))
        normalized_composition = {}
        for element, abundance in composition.items():
            if abundance != 0:
                normalized_composition[element] = abundance / total

        return normalized_composition

    def _construct_layer(
        self, altitude: u.Quantity, thickness_factor: float
    ) -> matter.Material:
        """
        Generates a Material object for the provided altitude.
        """

        altitude = altitude << u.km
        row = self._get_altitude_table_row(altitude)
        rho = row["total mass density"] << u.g / (u.cm**3)
        thickness = (
            np.diff(cast(u.Quantity, self.lookup_table["altitude"]))[0] * thickness_factor
        ) << u.cm

        elemental_abundances = self._compute_layer_composition(row)
        layer_attenuation = matter.Attenuations.from_compound_dict(elemental_abundances)
        return matter.Material(
            name="atmosphere",
            thickness=thickness,
            density=rho,
            attenuations=layer_attenuation,
            average_across_bins=False
        )

    def attenuate_spectrum_through_layers(
        self,
        flare_spectrum: flares.Flare,
        out_dir: str,
    ):
        """
        Attenuates the provided specturm from the maximum altitude down to
        the minimum altitude at the specified steps.

        Plots and pickle files are saved to a directory in out_dir.
        """

        dir_str = f"{flare_spectrum.goes_class}-layer-attenuation-zenith{self.solar_zenith.value}{self.solar_zenith.unit}"
        out_dir = os.path.join(out_dir, dir_str)
        plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        print(
            f"Iterating from {self.altitudes.max()} to {self.altitudes.min()} altitude"
        )

        thickness = thickness_through_zenith(
            self.solar_zenith,
            self.altitudes.max(),
            observer_altitude=self.altitudes.min(),
        )
        norm = self.altitudes.max() - self.altitudes.min()
        layer_thickness_factor = thickness / norm

        spectral_output = {"input": flare_spectrum, "layers": []}
        cumulative_transmission = np.ones(flare_spectrum.thermal.size)
        flare_spectrum = copy.deepcopy(flare_spectrum)

        for altitude in np.sort(self.altitudes, descending=True):
            print("Processing altitude", altitude)

            layer = self._construct_layer(altitude, layer_thickness_factor)
            layer.attenuations.photoelectric = self.photoelectric
            layer.attenuations.rayleigh = self.rayleigh
            layer.attenuations.compton = self.compton

            trans_vec = layer.compute_transmission(flare_spectrum.energy_edges)
            cumulative_transmission *= trans_vec
            spectral_output["layers"].append(
                (cast(float, altitude.to_value(u.km)), cumulative_transmission.copy())
            )

            ax = plot_spectrum(
                flare_spectrum.all_emission,
                flare_spectrum.energy_edges,
                flare_spectrum.goes_class,
                color="blue",
                label="layer incident spectrum",
            )

            flare_spectrum.all_emission *= trans_vec
            plot_spectrum(
                flare_spectrum.all_emission,
                flare_spectrum.energy_edges,
                flare_spectrum.goes_class,
                ax=ax,
                color="black",
                label="layer transmitted spectrum",
            )
            ax.set_title("Atmospheric attenuation")
            ax.legend()

            plot_file = os.path.join(
                plot_dir, f"{altitude.value}{altitude.unit}.png"
            )
            plt.savefig(plot_file, dpi=150)


        with open(os.path.join(out_dir, "transmissions.pkl"), "wb") as f:
            pickle.dump(spectral_output, f)


def plot_spectrum(
    spectrum: u.Quantity,
    energy_bins: u.Quantity,
    goes: str,
    ax: mpa.Axes | None = None,
    **kwargs,
) -> mpa.Axes:
    with plt.style.context(style_file):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6), layout="constrained")

        with quantity_support():  # pyright: ignore[reportGeneralTypeIssues]
            ax.stairs(spectrum, energy_bins, **kwargs)

        ax.set(
            xlabel="Energy (keV)",
            ylabel="ph / keV / sec / cm2",
            title=f"{goes} class spectrum",
            xscale="log",
            yscale="log",
            ylim=(1e-6, 1e9),
        )

        return ax
