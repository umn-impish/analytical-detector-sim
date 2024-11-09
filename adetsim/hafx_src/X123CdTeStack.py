import numpy as np
from ..sim_src.AttenuationData import AttenuationType, AttenuationData
from ..sim_src.DetectorStack import DetectorStack
from ..sim_src.FlareSpectrum import FlareSpectrum
from ..sim_src.Material import Material

from . import HafxMaterialProperties as hmp
from .detectors import LinearInterpolateDisperseDetector
from .X123Stack import X123Stack


class X123CdTeStack(DetectorStack):
    X123_DIAMETER = X123Stack.X123_DIAMETER  # cm
    DET_THICK = 1e-3 * hmp.METER_PER_CM  # cm
    BE_THICK = 100e-6 * hmp.METER_PER_CM  # cm

    E1 = 14.4  # keV
    E2 = 122  # keV
    FW1 = 530 / 1000  # keV
    FW2 = 850 / 1000  # keV

    def __init__(self, **kwargs):
        materials = [
            Material(
                diameter=self.X123_DIAMETER,
                attenuation_thickness=self.BE_THICK,
                mass_density=hmp.DENSITIES[hmp.BE],
                attenuation_data=AttenuationData.from_compound_dict(
                    hmp.ATTEN_FORMULAS[hmp.BE]
                ),
            )
        ]
        self.detector_volume = Material(
            diameter=self.X123_DIAMETER,
            attenuation_thickness=self.DET_THICK,
            mass_density=hmp.DENSITIES[hmp.CDTE],
            attenuation_data=AttenuationData.from_compound_dict(
                hmp.ATTEN_FORMULAS[hmp.CDTE]
            ),
        )
        super().__init__(
            materials,
            LinearInterpolateDisperseDetector(
                e1=self.E1,
                e2=self.E2,
                fwhm1=self.FW1 / self.E1,
                fwhm2=self.FW2 / self.E2,
            ),
        )

    def generate_detector_response_to(
        self,
        spectrum: FlareSpectrum,
        disperse_energy: bool,
        chosen_attenuations: list = AttenuationType.ALL,
    ) -> np.ndarray:
        prelim_resp = self._generate_material_response_due_to(
            spectrum, chosen_attenuations
        )
        abzorbed = np.ones(
            spectrum.energy_edges.size - 1
        ) - self.detector_volume.generate_overall_response_matrix_given(
            spectrum, [AttenuationType.PHOTOELECTRIC_ABSORPTION]
        )
        return self._dispatch_dispersion(
            spectrum, abzorbed * prelim_resp, disperse_energy
        )
