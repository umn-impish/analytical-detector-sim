import astropy.units as u
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from astropy.table import Row
from dataclasses import dataclass

import adetsim.sim_src.AttenuationData as ad
from adetsim.sim_src.FlareSpectrum import FlareSpectrum
from adetsim.sim_src.DetectorStack import DetectorStack
from adetsim.sim_src.Material import Material
from adetsim.hafx_src.Sipm3000 import Sipm3000
from .atmospheric_lookup_table import compute_lookup_table, ABUNDANCE_COLS

plt.style.use(os.path.join(os.path.dirname(__file__), "styles/plot.mplstyle"))


EARTH_RADIUS = 6378 * u.km


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


def generate_flare_spectrum(
    goes_class: str,
    min_energy: float = 2,
    max_energy: float = 300,
    energy_delta: float = 0.1,
) -> FlareSpectrum:
    """
    Energy parameters are in keV.
    energy_delta is the width of each energy bin.
    """

    edges = np.arange(min_energy, max_energy + energy_delta, energy_delta)
    fs = FlareSpectrum.make_with_battaglia_scaling(
        goes_class=goes_class, energy_edges=edges
    )

    return fs


def plot_spectrum(spectrum: FlareSpectrum, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")

    ax.stairs(spectrum.flare, spectrum.energy_edges, **kwargs)
    ax.set(
        xlabel="Energy (keV)",
        ylabel="ct / keV / sec / cm2",
        title=f"{spectrum.goes_class} class spectrum",
        xscale="log",
        yscale="log",
        ylim=(1e-6, 1e9),
    )

    return ax


@dataclass
class Layer:
    altitude: u.Quantity
    material: Material

    def attenuate_spectrum(
        self, input: FlareSpectrum, apply_energy_resolution: bool
    ) -> FlareSpectrum:
        self.input_spectrum = input
        detector = DetectorStack([self.material], Sipm3000())
        self.response = detector.generate_detector_response_to(
            input, disperse_energy=apply_energy_resolution
        )
        self.output_spectrum = self.response @ input.flare

        return self.output_spectrum


@dataclass
class Atmosphere:
    datetime: np.datetime64
    latitude: u.Quantity
    longitude: u.Quantity
    minimum_altitude: u.Quantity
    maximum_altitude: u.Quantity
    altitude_step: u.Quantity
    cross_section_diameter: u.Quantity = 10 * u.cm
    solar_zenith: u.Quantity = 0 * u.deg

    def __post_init__(self):
        self.lookup_table = compute_lookup_table(
            self.datetime,
            self.latitude,
            self.longitude,
            self.minimum_altitude,
            self.maximum_altitude,
            self.altitude_step,
        )

    def _get_altitude_table_row(self, altitude: u.Quantity) -> Row:
        """
        Helper function to retrieve data for a specific altitude.
        If the specified altitude is not in the table, this function
        returns the data for the altitude closest to that specified.
        """

        diffs = np.abs(self.lookup_table["altitude"] - altitude)
        index = np.where(diffs == diffs.min())[0][0]

        return self.lookup_table[index]

    def _compute_layer_composition(self, row: Row) -> dict[str, float]:
        """
        Computes the **elemental** abundances for the provided row.

        If there is a corresponding molecular form for the element
        (e.g. oxygen), then the composition accounts for the two
        oxygen atoms per molcule when computing the new abundances.
        """

        abundances = row[ABUNDANCE_COLS]
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

    def _construct_layer(self, altitude: u.Quantity, thickness_factor: float) -> Layer:
        """
        Generates a Material object for the provided altitude.
        """

        altitude = altitude << u.km
        row = self._get_altitude_table_row(altitude)
        rho = row["total mass density"] << u.g / (u.cm**3)
        thickness = (
            np.diff(self.lookup_table["altitude"])[0] * thickness_factor
        ) << u.cm

        elemental_abundances = self._compute_layer_composition(row)
        layer_attenuation = ad.AttenuationData.from_compound_dict(elemental_abundances)
        material = Material(
            self.cross_section_diameter.value,
            thickness.value,
            rho.value,
            layer_attenuation,
            name=f'atmosphere{row["altitude"].value}{row["altitude"].unit}',
        )
        layer = Layer(altitude, material)

        return layer

    def attenuate_spectrum_through_layers(
        self,
        flare_spectrum: FlareSpectrum,
        out_dir: str,
        apply_energy_resolution: bool = False,
    ) -> FlareSpectrum:
        """
        Attenuates the provided specturm from the maximum altitude down to
        the minimum altitude at the specified steps.

        Plots and pickle files are saved to a directory in out_dir.
        """

        # timestamp = Time.now().strftime('%Y%m%dT%H%M%S') # DO WE WANT TO TIMESTAMP THE DIRECTORIES?!?
        dir_str = f"{flare_spectrum.goes_class}-layer-attenuation-zenith{self.solar_zenith.value}{self.solar_zenith.unit}"
        out_dir = os.path.join(out_dir, dir_str)
        plot_dir = os.path.join(out_dir, "plots")
        pickle_dir = os.path.join(out_dir, "pickles")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(pickle_dir, exist_ok=True)

        print(
            f"Iterating from {self.maximum_altitude} to {self.minimum_altitude.value} altitude at {self.altitude_step} steps"
        )

        thickness = thickness_through_zenith(
            self.solar_zenith,
            self.maximum_altitude,
            observer_altitude=self.minimum_altitude,
        )
        norm = self.maximum_altitude - self.minimum_altitude
        layer_thickness_factor = thickness / norm

        layers = {}
        input_spectrum = copy.deepcopy(flare_spectrum)
        current_altitude = copy.deepcopy(self.maximum_altitude)
        while current_altitude >= self.minimum_altitude:
            print("Processing altitude", current_altitude)

            layer = self._construct_layer(current_altitude, layer_thickness_factor)
            layer_flare = layer.attenuate_spectrum(
                input_spectrum, apply_energy_resolution=apply_energy_resolution
            )
            layer_spectrum = FlareSpectrum(
                goes_class=input_spectrum.goes_class,
                energy_edges=input_spectrum.energy_edges,
                thermal=layer_flare,  # technically wrong, but it'll work
                nonthermal=np.zeros(shape=layer_flare.shape),
            )
            layers[current_altitude] = layer

            ax = plot_spectrum(
                input_spectrum, color="blue", label="layer incident spectrum"
            )
            plot_spectrum(
                layer_spectrum, ax=ax, color="black", label="layer output spectrum"
            )
            ax.set_title("Atmospheric attenuation")
            ax.legend()

            plot_file = os.path.join(
                plot_dir, f"{current_altitude.value}{current_altitude.unit}.png"
            )
            pickle_file = os.path.join(
                pickle_dir, f"{current_altitude.value}{current_altitude.unit}.pkl"
            )
            plt.savefig(plot_file, dpi=150)
            with open(pickle_file, "wb") as outfile:
                pickle.dump(layer_spectrum, outfile)

            current_altitude -= self.altitude_step
            input_spectrum = copy.deepcopy(layer_spectrum)
