"""
Contains functions for generating and plotting the atmospheric lookup table using Pymsis.
This table contains the elemental abundances at each altitude.
"""

import astropy.units as u
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.table import QTable
from pymsis import msis

plt.style.use(os.path.join(os.path.dirname(__file__), "styles/plot.mplstyle"))


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

    abundances = np.array([data[c].value for c in DENSITY_COLS])
    totals = np.sum(abundances, axis=0)
    abundances = abundances / totals
    data.add_columns(list(abundances), indexes=indices, names=abund_cols)


def compute_lookup_table(
    datetime: np.datetime64,
    lat: u.Quantity,
    lon: u.Quantity,
    min_altitude: u.Quantity,
    max_altitude: u.Quantity,
    step: u.Quantity,
    **run_kwargs,
) -> QTable:
    """
    Computes a lookup table for atmospheric composition using MSIS.
    """

    lat = lat << u.degree
    lon = lon << u.degree
    altitudes = (
        np.arange(
            (min_altitude << u.km).value,
            ((max_altitude + step) << u.km).value,
            (step << u.km).value,
        )
        * u.km
    )

    output = msis.run(
        dates=datetime,
        lons=lon.value,
        lats=lat.value,
        alts=altitudes.value,
        **run_kwargs,
    )
    output = np.squeeze(output)
    output[np.isnan(output)] = 0
    table = _convert_msis_output_to_qtable(output)
    table["altitude"] = altitudes
    table.meta = {"datetime": datetime, "lat": lat, "lon": lon}

    for c in (table.columns).copy():
        if c[-3:] == "den" and c not in DENSITY_COLS:
            table.remove_column(c)
            print("removed", c)

    _compute_abundances(table)

    return table


def plot_densities(data: QTable) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained", sharex=True)

    altitude = data["altitude"]
    marker = itertools.cycle(("x", "+", "o", "*", "v", "s"))
    for column in DENSITY_COLS:
        if "den" in column and isinstance(data[column], u.Quantity):
            label = column.replace("den", "")
            ax.plot(
                altitude,
                data[column],
                marker=next(marker),
                ls="--",
                markersize=6,
                lw=0.25,
                label=label,
            )
    ax.plot(
        altitude,
        data["total mass density"],
        marker=next(marker),
        ls="--",
        markersize=3,
        lw=0.25,
        label="air",
    )
    ax.axvline(40, c="gray", ls=":")

    ax.set(
        xlabel=f"Altitude ({altitude.unit})",
        ylabel=f"Density ({data[column].unit})",
        yscale="log",
        ylim=(1e-20, ax.get_ylim()[1]),
    )
    ax.legend()

    return ax


def plot_abundances(data: QTable, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained", sharex=True)

    altitude = data["altitude"]

    for column in data.columns:
        if "abund" in column:
            label = column.replace("abund", "")
            ax.plot(
                altitude,
                data[column],
                marker=next(MARKERS),
                ls="--",
                markersize=6,
                lw=0.3,
                label=label,
            )
    ax.axvline(40, c="gray", ls=":")

    ax.set(
        xlabel=f"Altitude ({altitude.unit})",
        ylabel="Abundance",
        title=f'Atmospheric Abundances on {data.meta["datetime"]}, {data.meta["lat"]}, {data.meta["lon"]}',
        yscale="log",
        xlim=(altitude.min().value, altitude.max().value),
        ylim=(1e-5, 1),
    )
    ax.legend()

    return ax


def plot_abundances_stackplot(data: QTable, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    abund_cols = [c.replace("den", "") + "abund" for c in DENSITY_COLS]
    labels = [c.replace("abund", "") for c in abund_cols]
    y = np.array([list(r.values()) for r in data[abund_cols]]).T * 100
    altitude = data["altitude"]

    ax.stackplot(altitude.value, y, labels=labels)

    ax.set(
        xlabel=f"Altitude ({altitude.unit})",
        ylabel="Abundance",
        title=f'Atmospheric Abundances on {data.meta["datetime"]}, {data.meta["lat"]}, {data.meta["lon"]}',
        yscale="log",
        xlim=(altitude.min().value, altitude.max().value),
        ylim=(75, 100),
    )
    ax.legend()

    return ax
