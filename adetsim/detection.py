import copy

import numpy as np
from astropy import units as u

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