import copy
from typing import cast

import numpy as np
from astropy import units as u
from scipy import integrate, interpolate

from . import material_manager as mman


class Attenuations:
    ATTENUATION_TYPES = ("photoelectric", "rayleigh", "compton")

    @classmethod
    def from_element(cls, element: str):
        """Construct an Attenuations for a single element."""
        return cls(coefficients={element: mman.fetch_element(element)})

    @classmethod
    def from_compound_dict(cls, compound: dict[str, float]):
        """Construct an Attenuations using the same compound format as
        in `material_manager`."""
        return cls(coefficients=mman.fetch_compound(compound))

    def __init__(self, coefficients: dict[str, dict[str, u.Quantity]]):
        """Construct an Attenuations management object with an attenuation data dict."""
        self.energies: dict[str, u.Quantity] = {}
        self.coefs: dict[str, dict[str, u.Quantity]] = {}
        for k, data in coefficients.items():
            data = copy.deepcopy(data)
            self.energies[k] = data.pop("energy")
            self.coefs[k] = {}
            for type in data:
                self.coefs[k][type] = copy.deepcopy(data[type])

        # By default, only enable photoelectric absorption
        self.photoelectric = True
        self.compton = False
        self.rayleigh = False
        self.setup_interpolators()

    def setup_interpolators(self):
        """Constructs a set of functions which interpolate log(attenuation coeff)
        data (to be integrated or evaluated), one per element
        """
        interp_log_att_funcs = {k: [] for k in Attenuations.ATTENUATION_TYPES}
        for name, data in self.coefs.items():
            energies = self.energies[name].to_value(u.keV)
            for key, att in data.items():
                # interpolate between NIST energies
                # we want straight-line interpolation on the log plot,
                # so take the log before doing any fitting
                loge, logat = np.log(energies), np.log(att.to_value(u.cm**2 / u.g))
                interp_func = interpolate.interp1d(
                    x=loge,
                    y=logat,
                    fill_value="extrapolate",  # pyright: ignore[reportArgumentType]
                )
                interp_log_att_funcs[key].append(interp_func)

        self.log_interpolators = interp_log_att_funcs

    def log_interpolate(self, att_type: str, log_energies: np.ndarray) -> np.ndarray:
        """
        Log-interpolate a given attenuation type to a set of energies.
        The mass-attenuation coefficients span multiple orders of magnitude across both energy and value,
        so to interpolate them properly, both the energy and values must be interpolated logarithmically.

        The coefficients are exponentiated back to physical units after the logarithmic interpolation is performed.

        NB: units are assumed to be cm, g, keV, where applicable.
        Use __call__ convenience method if you want quantities.
        """
        ret = np.zeros_like(log_energies)
        for f in self.log_interpolators[att_type]:
            ret += np.exp(f(log_energies))
        return ret

    def unitless_interpolate(self, energy_points: np.ndarray):
        """Same as __call__, but no units."""
        ret = np.zeros(energy_points.shape)
        log_energies = np.log(energy_points)
        if self.photoelectric:
            ret += self.log_interpolate(
                att_type=Attenuations.ATTENUATION_TYPES[0], log_energies=log_energies
            )
        if self.rayleigh:
            ret += self.log_interpolate(
                att_type=Attenuations.ATTENUATION_TYPES[1], log_energies=log_energies
            )
        if self.compton:
            ret += self.log_interpolate(
                att_type=Attenuations.ATTENUATION_TYPES[2], log_energies=log_energies
            )
        return ret

    @u.quantity_input
    def __call__(self, energy_points: u.keV):  # pyright: ignore[reportInvalidTypeForm]
        """Evaluate the turned-on attenuation types at a set of energy **points** (not bins)."""
        return self.unitless_interpolate(energy_points.to_value(u.keV)) << (
            u.cm**2 / u.g
        )

    def __repr__(self):
        ident = "|".join(k for k in self.coefs)
        type_present = [self.photoelectric, self.rayleigh, self.compton]
        type_str = " + ".join(
            t for (t, yn) in zip(Attenuations.ATTENUATION_TYPES, type_present) if yn
        )
        return f"Attenuations[{ident}, {type_str}]"


class Material:
    @u.quantity_input
    def __init__(
        self,
        name: str,
        thickness: u.mm,  # pyright: ignore[reportInvalidTypeForm]
        density: (u.g / u.cm**3),  # pyright: ignore[reportInvalidTypeForm]
        attenuations: Attenuations,
        average_across_bins: bool,
    ):
        self.name = name
        self._thickness: float = thickness.to_value(u.cm)
        self._density: float = density.to_value(u.g / u.cm**3)
        self.attenuations = attenuations
        self.average_across_bins = average_across_bins

    @property
    def thickness(self):
        return self._thickness << u.cm

    @thickness.setter
    def thickness(self, other: u.Quantity):
        self._thickness = other.to_value(u.cm)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def density(self):
        return self._density << (u.g / u.cm**3)

    @density.setter
    def density(self, other: u.Quantity):
        self._density = other.to_value(u.g / u.cm**3)  # pyright: ignore[reportAttributeAccessIssue]

    @u.quantity_input
    def compute_transmission(
        self, energy_edges: u.Quantity[u.keV]
    ) -> u.Quantity[u.one]:
        if energy_edges.ndim > 1:
            raise ValueError("Energy edges must be 1D array")

        if not self.average_across_bins:
            mids = energy_edges[:-1] + (energy_edges[1:] - energy_edges[:-1]) / 2
            return np.exp(-1 * self.attenuations(mids) * self.density * self.thickness)

        def transmission(e):
            return np.exp(
                -1
                * self.attenuations.unitless_interpolate(np.atleast_1d(e))
                * self._density
                * self._thickness
            )[0]

        # average the interaction probabilities across a bin via integration
        ret = np.zeros(energy_edges.size - 1)
        dimensionless_edges = cast(np.ndarray, energy_edges.to_value(u.keV))
        for i in range(ret.size):
            integrated_transm_prob, _ = integrate.quad(
                func=transmission,
                a=dimensionless_edges[i],
                b=dimensionless_edges[i + 1],
            )
            avg_transm_prob = integrated_transm_prob / (
                dimensionless_edges[i + 1] - dimensionless_edges[i]
            )
            ret[i] = avg_transm_prob

        return ret << u.one

    @u.quantity_input
    def compute_absorption(self, energy_edges: u.keV) -> u.one:  # pyright: ignore[reportInvalidTypeForm]
        return np.ones(energy_edges.size - 1) - self.compute_transmission(
            energy_edges=energy_edges
        )

    def __repr__(self):
        ret = f"<Material {self.name}, {self.density:.2f}, "
        ret += f"{self.thickness:.2e} thick>"
        return ret
