import numpy as np
import os

from ..sim_src.FlareSpectrum import FlareSpectrum
from .HafxStack import HafxStack
from .HafxMaterialProperties import SINGLE_DET_AREA


class HafxSimulationContainer:
    MIN_ENG = 1.0  # keV
    MAX_ENG = 300.0  # keV
    DE = 0.1  # keV
    MIN_THRESHOLD_ENG = MIN_ENG  # keV
    MAX_THRESHOLD_ENG = MAX_ENG  # keV

    KMINIMAL = "minimal"
    KAL_THICKNESS = "al_thickness"
    KGOES_CLASS = "goes_class"
    KMINE = "min_e"
    KMAXE = "max_e"
    KDE = "de"

    KENERGY_EDGES = "energy_edges"
    KFLARE_THERMAL = "thermal"
    KFLARE_NONTHERMAL = "nonthermal"

    KPURE_RESPONSE = "pure_response_matrix"
    KDISPERSED_RESPONSE = "dispersed_response_matrix"
    MATRIX_KEYS = (KDISPERSED_RESPONSE, KPURE_RESPONSE)

    DEFAULT_SAVE_DIR = "responses-and-areas"

    @classmethod
    def from_file(cls, filename: str, remake_spectrum=False):
        """alias"""
        return cls.from_saved_file(filename, remake_spectrum=remake_spectrum)

    @classmethod
    def from_saved_file(cls, filename: str, remake_spectrum=False):
        """load the container from a (compressed) .npz file"""
        data = np.load(filename, allow_pickle=True)
        goes_class = str(data[cls.KGOES_CLASS])

        if remake_spectrum:
            edges = data[cls.KENERGY_EDGES]
            fs = FlareSpectrum.make_with_battaglia_scaling(
                goes_class=goes_class, energy_edges=edges
            )

        else:
            fs = FlareSpectrum(
                goes_class=goes_class,
                thermal=data[cls.KFLARE_THERMAL],
                nonthermal=data[cls.KFLARE_NONTHERMAL],
                energy_edges=data[cls.KENERGY_EDGES],
            )

        ret = cls(aluminum_thickness=data[cls.KAL_THICKNESS], flare_spectrum=fs)
        for k in cls.MATRIX_KEYS:
            ret.matrices[k] = data[k]
        return ret

    def __init__(
        self,
        aluminum_thickness: np.float64 = None,
        flare_spectrum: FlareSpectrum = None,
    ):
        self.detector_stack = HafxStack()
        self.al_thick = aluminum_thickness
        self.matrices = {k: None for k in self.MATRIX_KEYS}
        self.flare_spectrum = flare_spectrum

    @property
    def al_thick(self):
        return self.__al_thick

    @al_thick.setter
    def al_thick(self, new):
        self.__al_thick = new
        self.detector_stack.materials[0].thickness = new

    @property
    def goes_class(self):
        return self.flare_spectrum.goes_class

    def compute_effective_area(
        self, cps_threshold: np.int64 = 0, different_flare: FlareSpectrum = None
    ):
        fspec_of_interest = different_flare or self.flare_spectrum
        if cps_threshold > 0:
            restrict = np.logical_and(
                fspec_of_interest.energy_edges >= self.MIN_THRESHOLD_ENG,
                fspec_of_interest.energy_edges <= self.MAX_THRESHOLD_ENG,
            )
            try:
                dispersed_flare = np.matmul(
                    self.matrices[self.KDISPERSED_RESPONSE], fspec_of_interest.flare
                )
            except ValueError:
                print(self.matrices[self.KDISPERSED_RESPONSE])
                raise
            bin_widths = np.diff(fspec_of_interest.energy_edges[restrict])
            relevant_cps = np.sum(
                dispersed_flare[restrict[:-1]] * SINGLE_DET_AREA * bin_widths
            )

            # "set" effective area to zero if we get more than the threshold counts
            if relevant_cps > cps_threshold:
                return np.zeros(fspec_of_interest.energy_edges.size - 1)

        area_vector = np.ones_like(fspec_of_interest.flare) * SINGLE_DET_AREA
        att_area = self.matrices[self.KPURE_RESPONSE] @ area_vector
        return att_area

    def simulate(self, other_spectrum: FlareSpectrum = None):
        fs = other_spectrum or self.flare_spectrum
        if self.al_thick is None:
            raise ValueError("Aluminum thickness has not been set.")
        # get the un-dispersed (pure) response matrix
        self.matrices[self.KPURE_RESPONSE] = (
            self.detector_stack.generate_detector_response_to(fs, False)
        )
        # apply CeBr3 energy resolution
        self.matrices[self.KDISPERSED_RESPONSE] = (
            self.detector_stack.apply_detector_dispersion_for(
                fs, self.matrices[self.KPURE_RESPONSE]
            )
        )

    def gen_file_name(self, prefix):
        gc = self.flare_spectrum.goes_class
        return f"{prefix or 'no_prefix'}_{gc or 'no_goes'}_{self.al_thick:.3e}cm_hafx"

    def save_to_file(self, out_dir=DEFAULT_SAVE_DIR, prefix=None):
        """save object data into a file that can be loaded back in later"""
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if None in self.matrices:
            raise ValueError("Matrices haven't been computed so we can't save them.")
        if self.flare_spectrum is None:
            raise ValueError(
                "Stored FlareSpectrum is None--simulation probably hasn't run."
            )

        to_save = dict()
        to_save[self.KGOES_CLASS] = self.flare_spectrum.goes_class
        to_save[self.KAL_THICKNESS] = self.al_thick

        to_save[self.KFLARE_THERMAL] = self.flare_spectrum.thermal
        to_save[self.KFLARE_NONTHERMAL] = self.flare_spectrum.nonthermal
        to_save[self.KENERGY_EDGES] = self.flare_spectrum.energy_edges

        for k in self.MATRIX_KEYS:
            to_save[k] = self.matrices[k]
        outfn = os.path.join(out_dir, self.gen_file_name(prefix))
        np.savez_compressed(outfn, **to_save)
