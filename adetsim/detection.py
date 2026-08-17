import copy
import functools
from collections.abc import Callable

import numpy as np
from astropy import units as u
from scipy import optimize as sco
from scipy.integrate import quad
from scipy.special import erf

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
    def generate_resolution_matrix(self, energy_bins: u.Quantity[u.keV]) -> np.ndarray:
        r"""Generate the energy resolution for an instrument assuming `1 / sqrt(E)` scaling.
        The pivot energies are set in the constructor along with their FWHMs.

        The resulting matrix is a collection of *row* vectors whose entries are pixellated
        Gaussians, such that the sum of the entire row of (infinite) pixels is 1.
        Each pixel is a probability element.
        The sum of each row need not be 1 if the starting pixel is near the edge.
        This is equivalent to saying, "We can't measure energies above or below our measurement range."

        To multiply the matrix onto an energy redistribution matrix, do (resolution matrix) @ (redistribution matrix).
        This multiplication is (should be?) equivalent to convolving the pixellated Gaussians.
        The response matrix should have its off-diagonal elements in upper rows of the matrix,
        that is, row indices <= column indices.
        This is the standard notation. Keep in mind that Matplotlib displays such matrices flipped
        when using e.g. `pcolormesh`, so the lower indices correspond to lower x or y values on the plot.
        """
        def smear_integral(start, end, mu, fwhm):
            """The probability integral for a Gaussian across a bin defined by
            `[start, end]` centered on `mu` with a FWHM `fwhm`.
            
            The error function is defined as the integral across (0 --> x),
            so subtracting two evaluations gives the definite integral of a Gaussian
            function between two bounds.
            """
            s = fwhm / 2 / np.sqrt(2 * np.log(2))
            left = (start - mu) / s
            right = (end - mu) / s

            # The factor of (1 / 2) is because `erf` is normalized
            # s.t. erf(inf) = 1, but the probability function is normalized
            # s.t. integral(-inf, inf) = 1, i.e. half the total area of erf.
            return (1 / 2) * (erf(right) - erf(left))

        bins = np.asarray(energy_bins.to_value(u.keV))
        mids = bins[:-1] + (de := np.diff(bins)) / 2
        fwhms = self.resolution_function(mids)
        ret = np.empty((mids.size, mids.size))

        for i in np.arange(mids.size):
            mid = mids[i]
            # The FWHMs are given as percentages, so convert them back to keV
            width = fwhms[i] * mid

            # We need to apply the gaussian centered at the energy midpoint
            # of the current row with the FWHM computed for that energy
            this_integ = functools.partial(smear_integral, mu=mid, fwhm=width)

            for j in np.arange(mids.size):
                # Now, for each column, sum the probability flux
                # redistributed from the diagonal pixel into the current energy bin
                ret[i][j] = this_integ(mids[j] - de[j] / 2, mids[j] + de[j] / 2)

        return ret
