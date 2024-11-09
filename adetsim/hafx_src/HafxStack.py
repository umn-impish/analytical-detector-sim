import numpy as np

from ..sim_src.AttenuationData import AttenuationData, AttenuationType
from ..sim_src.DetectorStack import DetectorStack
from ..sim_src.FlareSpectrum import FlareSpectrum
from ..sim_src.Material import Material

from . import detectors

from .HafxMaterialProperties import (
    HAFX_MATERIAL_ORDER,
    AL,
    THICKNESSES,
    DENSITIES,
    ATTEN_FORMULAS,
    DIAMETER,
)


def gen_materials(att_thick: np.float64):
    """put the HaFX materials in the right order (variable aluminum thickness)"""
    mat_order = HAFX_MATERIAL_ORDER
    materials = []
    for name in mat_order:
        thick = att_thick if name == AL else THICKNESSES[name]
        rho = DENSITIES[name]
        atten_dat = AttenuationData.from_compound_dict(ATTEN_FORMULAS[name])
        materials.append(Material(DIAMETER, thick, rho, atten_dat, name=name))
    return materials


class HafxStack(DetectorStack):
    """photoabsorption into the scintillator crystal is different here so we need separate behavior."""

    def __init__(
        self, enable_scintillator: bool = True, att_thick: np.float64 = NotImplemented
    ):
        # Measured resolution: 60% at 20 keV
        det = detectors.SqrtEnergyResolutionDetector(0.6, 20)
        super().__init__(gen_materials(att_thick), det)

        # take off the scintillator to treat it separately
        self.scintillator = self.materials.pop()
        # XXX: set to False to disable the scintillator (i.e. only disperse spectrum, dont absorb it)
        self.enable_scintillator = enable_scintillator

    @property
    def att_thick(self):
        return self.materials[0].thickness

    @att_thick.setter
    def att_thick(self, new):
        self.materials[0].thickness = new

    def generate_detector_response_to(
        self,
        incident_spectrum: FlareSpectrum,
        disperse_energy: bool,
        chosen_attenuations: list = AttenuationType.ALL,
    ) -> np.ndarray:
        response = self._generate_material_response_due_to(
            incident_spectrum, chosen_attenuations
        )
        if self.enable_scintillator:
            # now incorporate the scintillator
            absorbed = self.generate_scintillator_response(
                incident_spectrum, chosen_attenuations
            )
            response = absorbed @ response
        return self._dispatch_dispersion(incident_spectrum, response, disperse_energy)

    def generate_scintillator_response(
        self, incident_spectrum: FlareSpectrum, chosen_attenuations: list
    ) -> np.ndarray:
        onez = np.ones(incident_spectrum.energy_edges.size - 1)
        # we must include photoelectric absorption as this mechanism leads to scintillation.
        abs_atts = list(
            set([AttenuationType.PHOTOELECTRIC_ABSORPTION] + chosen_attenuations)
        )
        # XXX: only dimensions of incident_spectrum used in this call. confusing...
        absorbed = onez - self.scintillator.generate_overall_response_matrix_given(
            incident_spectrum, abs_atts
        )
        return np.diag(absorbed)
