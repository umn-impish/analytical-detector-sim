import numpy as np

from adetsim.sim_src import AttenuationData as ad
from adetsim.sim_src import DetectorStack as ds
from adetsim.sim_src import FlareSpectrum as fs
from adetsim.sim_src import Material as mat
from adetsim.hafx_src.detectors import FixedEnergyResolutionDetector
from adetsim.hafx_src import HafxMaterialProperties as hmp

MAT_FORMULAS = {
    "teflon": {"C": 2, "F": 4},
    "pla": {"C": 3, "H": 4, "O": 2},
    "gagg": {"Gd": 2.9995, "Ce": 0.0005, "Al": 2, "Ga": 5, "O": 12},
    "lyso": {"Lu": 1.98, "Y": 0.02, "Si": 1, "O": 5},
}

MAT_DAT = {
    k: ad.AttenuationData.from_compound_dict(d) for (k, d) in MAT_FORMULAS.items()
}

MAT_DENS = {
    "teflon": 2.2,  # g/cm3
    "pla": 1.24,  # g/cm3
    "lyso": 7.1,  # g/cm3
    "gagg": 6.63,  # g/cm3
}


class GripsStack(ds.DetectorStack):
    # We have 4 x 4cm2 squares.
    # The equivalent circle has radius sqrt(16 / pi) cm
    # Or diameter of 2r = 2 * 2.26 = 4.52 cm2.
    DIAMETER = 4.52  # cm

    def __init__(self, stack_mats: dict[str, float]):
        det_mats = ("lyso", "gagg")
        self.detector = None
        mats = []
        for n, t in stack_mats.items():
            kw = dict(
                diameter=GripsStack.DIAMETER,
                attenuation_thickness=t,
                mass_density=MAT_DENS[n] if n in MAT_DENS else hmp.DENSITIES[n],
                attenuation_data=MAT_DAT[n]
                if n in MAT_DAT
                else ad.AttenuationData.from_compound_dict(hmp.ATTEN_FORMULAS[n]),
                name=n,
            )
            if n not in det_mats:
                m = mat.Material(**kw)
                mats.append(m)
                continue
            self.detector = mat.Material(**kw)

        res = 0.3
        super().__init__(mats, FixedEnergyResolutionDetector(res))

    def generate_detector_response_to(
        self,
        spectrum: fs.FlareSpectrum,
        disperse_energy: bool,
        chosen_attenuations: tuple = ad.AttenuationType.ALL,
    ) -> np.ndarray:
        prelim_resp = self._generate_material_response_due_to(
            spectrum, chosen_attenuations
        )
        abzorbed = np.ones(
            spectrum.energy_edges.size - 1
        ) - self.detector.generate_overall_response_matrix_given(
            spectrum, [ad.AttenuationType.PHOTOELECTRIC_ABSORPTION]
        )
        return self._dispatch_dispersion(
            spectrum, abzorbed * prelim_resp, disperse_energy
        )
