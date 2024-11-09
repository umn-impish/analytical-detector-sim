import numpy as np
from scipy import integrate
from .AttenuationData import AttenuationData, AttenuationType
from .FlareSpectrum import FlareSpectrum


class Material:
    def __init__(
        self,
        diameter: np.float64,
        attenuation_thickness: np.float64,
        mass_density: np.float64,
        attenuation_data: AttenuationData,
        name: str = None,
    ):
        self.name = name or "(no name given)"
        self.diameter = diameter
        self.mass_density = mass_density
        self.attenuation_data = attenuation_data
        self.thickness = attenuation_thickness

    @property
    def area(self):
        return (self.diameter / 2) ** 2 * np.pi

    def generate_overall_response_matrix_given(
        self, incident_spectrum: FlareSpectrum, attenuations: list
    ) -> np.ndarray:
        dim = incident_spectrum.energy_edges.size - 1
        vec = np.ones(dim, dtype=np.float64)
        # everything is diagonal except Compton scattering, so the matrices commute
        # i.e. multiplication order doesn't matter
        for k in attenuations:
            vec *= self.generate_modifying_matrix_for(k, incident_spectrum)
        return vec

    def generate_modifying_matrix_for(
        self, mechanism_type, incident_spectrum: FlareSpectrum
    ) -> np.ndarray:
        modify_gen_lookup = {
            AttenuationType.PHOTOELECTRIC_ABSORPTION: self._gen_photo,
            AttenuationType.RAYLEIGH: self._gen_rayleigh,
            AttenuationType.COMPTON: self._gen_compton,
        }
        return modify_gen_lookup[mechanism_type](incident_spectrum)

    def _gen_phot_ray(self, which, incident_spectrum: FlareSpectrum) -> np.ndarray:
        # Log attenuation function (to interpolaetaeaeae)
        log_att = lambda e: self.attenuation_data.log_interpolate(which, e)

        def mass_att_exponent(log_energy):
            return -1 * np.exp(log_att(log_energy)) * self.mass_density * self.thickness

        def transm_prob(energy):
            return np.exp(mass_att_exponent(np.log(energy)))

        # average the interaction probabilities across a bin via integration
        ret = np.zeros(incident_spectrum.flare.size)
        edges = incident_spectrum.energy_edges
        for i in range(edges.size - 1):
            integrated_transm_prob, _ = integrate.quad(
                transm_prob, edges[i], edges[i + 1]
            )
            avg_transm_prob = integrated_transm_prob / (edges[i + 1] - edges[i])
            ret[i] = avg_transm_prob

        return ret

    def _gen_photo(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        return self._gen_phot_ray(
            AttenuationType.PHOTOELECTRIC_ABSORPTION, incident_spectrum
        )

    def _gen_rayleigh(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        return self._gen_phot_ray(AttenuationType.RAYLEIGH, incident_spectrum)

    def _gen_compton(self, incident_spectrum: FlareSpectrum) -> np.ndarray:
        """
        NB: not implemented!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        # do nothing
        return np.ones(incident_spectrum.energy_edges.size - 1)

    def gen_char_xrays(self, photoelec_att: np.ndarray, inc: FlareSpectrum):
        raise NotImplementedError

    def __repr__(self):
        ret = f"<Material {self.name}, {self.mass_density:.2f} g/cm3, "
        ret += f"{self.diameter:.2e} cm diameter, {self.thickness:.2e} cm thick>"
        return ret
