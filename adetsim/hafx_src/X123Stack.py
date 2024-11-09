import numpy as np
import astropy.units as u

from ..sim_src.AttenuationData import AttenuationType, AttenuationData
from ..sim_src.DetectorStack import DetectorStack
from ..sim_src.FlareSpectrum import FlareSpectrum
from ..sim_src.Material import Material

from . import HafxMaterialProperties as hmp
from . import detectors


# The Al window thickness we decided on
WINDOW_THICK = (10 << u.um).to_value(u.cm)


class X123Stack(DetectorStack):
    """material stack for X-123 detector from AmpTek"""

    # Collimated diameter
    X123_DIAMETER = 2 * np.sqrt(0.17 / np.pi)  # cm
    DET_THICK = (1000 << u.um).to_value(u.cm)
    BE_THICK = (8 << u.um).to_value(u.cm)

    def __init__(
        self, det_thick: float = DET_THICK, window_thickness: float = WINDOW_THICK
    ):
        # Aluminum window in front of Be filter
        materials = [
            Material(
                diameter=self.X123_DIAMETER,
                attenuation_thickness=window_thickness,
                mass_density=hmp.DENSITIES[hmp.AL],
                attenuation_data=AttenuationData.from_compound_dict(
                    hmp.ATTEN_FORMULAS[hmp.AL]
                ),
                name="Al",
            ),
            Material(
                diameter=self.X123_DIAMETER,
                attenuation_thickness=self.BE_THICK,
                mass_density=hmp.DENSITIES[hmp.BE],
                attenuation_data=AttenuationData.from_compound_dict(
                    hmp.ATTEN_FORMULAS[hmp.BE]
                ),
                name="Be",
            ),
        ]
        self.detector_volume = Material(
            diameter=self.X123_DIAMETER,
            attenuation_thickness=det_thick,
            mass_density=hmp.DENSITIES[hmp.SI],
            attenuation_data=AttenuationData.from_compound_dict(
                hmp.ATTEN_FORMULAS[hmp.SI]
            ),
        )

        super().__init__(
            materials,
            # At 6 keV we assume 3% FWHM, and that it's approx
            # constant across the energy range
            detectors.SqrtEnergyResolutionDetector(resolution_pct=0.03, energy=6.0),
        )

    def generate_detector_response_to(
        self,
        spectrum: FlareSpectrum,
        disperse_energy: bool,
        chosen_attenuations: tuple = AttenuationType.ALL,
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
