import copy
import functools
from collections.abc import Callable

import numpy as np
from astropy import units as u
from scipy import optimize as sco
from scipy.integrate import quad

from . import matter


class PhysicalDetectorStack:
    """Absorb X-rays in a final layer of a material stack.
    Things like energy resolution, etc. are captured further down the line."""

    def __init__(self, materials: list[matter.Material]):
        self.materials = copy.deepcopy(materials)

    @u.quantity_input
    def compute_absorption(self, energy_edges: u.keV) -> u.one:  # pyright: ignore[reportInvalidTypeForm]
        ret = np.ones(energy_edges.size - 1)
        for m in self.materials[:-1]:
            ret *= m.compute_transmission(energy_edges=energy_edges)
        return self.materials[-1].compute_absorption(energy_edges=energy_edges) * ret


# TODO resolution etc.
class SqrtEnergyResolution:
    @u.quantity_input
    def __init__(
        self,
        reference_fwhms: u.Quantity[u.percent],
        fwhm_errors: u.Quantity[u.percent],
        reference_energies: u.Quantity[u.keV],
    ):
        self.fwhms = reference_fwhms
        self.anchor_energies = reference_energies

        self.resolution_function: Callable[[np.ndarray], np.ndarray]
        self._fit_resolution_function(fwhm_errors)

    @u.quantity_input
    def _fit_resolution_function(self, errors: u.Quantity[u.percent]):
        r"""Fit a function of the form $\alpha / \sqrt{E}$ to the provided FWHMs and anchor energies"""

        def resolution_func(e, scale):
            return scale / np.sqrt(e)

        anchors = np.asarray(self.anchor_energies.to_value(u.keV))
        fwhms = np.asarray(self.fwhms.to_value(u.one))
        sigma = np.asarray(errors.to_value(u.one))
        (scale,), _ = sco.curve_fit(resolution_func, anchors, fwhms, sigma=sigma)
        self.resolution_function = functools.partial(resolution_func, scale=scale)

    @u.quantity_input
    def generate_resolution_matrix(
        self, energy_bins: u.Quantity[u.keV], cut: float
    ) -> np.ndarray:
        """Generate the energy resolution for an instrument assuming 1 / sqrt(E) scaling.
        The pivot energies are set in the constructor along with their FWHMs.

        The `cut` parameter indicates how small the probability bin may be before it is truncated.
        """
        integral_scale = 1 / np.sqrt(np.pi)

        def smear(e, mu, fwhm):
            """A Gaussian function normalized on [-inf, inf]"""
            s = fwhm / 2 / np.log(2)
            return (integral_scale / s) * np.exp(-(e - mu) * (e - mu) / (s * s))

        bins = np.asarray(energy_bins.to_value(u.keV))
        mids = bins[:-1] + (de := np.diff(bins)) / 2
        fwhms = self.resolution_function(mids)
        ret = np.empty((mids.size, mids.size))
        for i in np.arange(mids.size):
            mid = mids[i]
            width = fwhms[i] * mid
            this_smear = functools.partial(smear, mu=mid, fwhm=width)
            for j in np.arange(mids.size):
                this_mid = mids[j]
                res, *_ = quad(this_smear, this_mid - de[j] / 2, this_mid + de[j] / 2)
                ret[i][j] = res if res > cut else 0

        return ret
