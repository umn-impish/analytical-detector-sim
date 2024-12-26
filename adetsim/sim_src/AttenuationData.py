import astropy.units as u
import numpy as np
from scipy import interpolate

from . import material_manager as mman


class AttenuationType:
    PHOTOELECTRIC_ABSORPTION = 1
    RAYLEIGH = 2
    COMPTON = 3
    ALL = [PHOTOELECTRIC_ABSORPTION, RAYLEIGH, COMPTON]

    @staticmethod
    def named():
        return {
            k: n
            for (k, n) in zip(
                AttenuationType.ALL, ("photoelectric", "rayleigh", "compton")
            )
        }


class AttenuationData:
    @classmethod
    def from_compound_dict(cls, compound: dict[str, float]):
        '''
        Construct an AttenuationData using the same compound format as
        in `material_manager`.
        '''
        weighted_coeffs = mman.fetch_compound(compound)
        for (name, coeffs) in weighted_coeffs.items():
            clean = {
                'energy': (coeffs['energy'] << u.MeV).to_value(u.keV),
            }
            for k in ('rayleigh', 'compton', 'photoelectric'):
                clean[k] = (coeffs[k] << u.cm**2 / u.g).to_value(u.cm**2 / u.g)
            weighted_coeffs[name] = clean
        return AttenuationData(weighted_coeffs)

    def __init__(self, coefficients: dict[str, dict[str, np.ndarray]]):
        """Do not call this directly."""
        self.energies = dict()
        self.attenuations = dict()
        for k, data in coefficients.items():
            self.energies[k] = data["energy"]
            if k not in self.attenuations:
                self.attenuations[k] = dict()

            self.attenuations[k][AttenuationType.PHOTOELECTRIC_ABSORPTION] = data[
                "photoelectric"
            ]
            self.attenuations[k][AttenuationType.RAYLEIGH] = data["rayleigh"]
            self.attenuations[k][AttenuationType.COMPTON] = data["compton"]

        self.setup_interpolators()

    def setup_interpolators(self):
        """returns a function that interpolates log of attenuation
        data (to be integrated or evaluated),
        one per element
        """
        interp_log_att_funcs = {k: [] for k in AttenuationType.ALL}
        for name, data in self.attenuations.items():
            energies = self.energies[name]
            for key, att in data.items():
                # interpolate between NIST energies
                # we want straight-line interpolation on the log plot,
                # so take the log before doing any fitting
                loge, logat = np.log(energies), np.log(att)
                interp_func = interpolate.interp1d(
                    x=loge, y=logat, fill_value="extrapolate"
                )
                interp_log_att_funcs[key].append(interp_func)

        self.log_interpolators = interp_log_att_funcs

    def log_interpolate(self, att_type, log_energies) -> np.ndarray:
        """
        Log-interpolate a given attenuation type to a set of log energies.
        """
        ret = np.zeros_like(log_energies)
        for f in self.log_interpolators[att_type]:
            ret += np.exp(f(log_energies))
        return np.log(ret)
