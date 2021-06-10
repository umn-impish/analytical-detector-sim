import numpy as np
from FlareSpectrum import FlareSpectrum

class AttenuationType:
    PHOTOELECTRIC_ABSORPTION = 1
    RAYLEIGH = 2
    COMPTON = 3
    ALL = [PHOTOELECTRIC_ABSORPTION, RAYLEIGH, COMPTON]

class AttenuationData:
    @classmethod
    def from_nist_file(cls, file_path: str):
        '''
        assumes:
            - tab-delimited
            - energy first col
            - photoelectric attenuation second col
            - rayleigh third col
            - compton fourth col
        '''
        # "unpack" transposes the file contents
        energies, photo, rayleigh, compton = np.loadtxt(file_path, dtype=np.float64, unpack=True)
        return cls(energies, photo, rayleigh, compton)

    def __init__(self, energies: np.ndarray, photoelec: np.ndarray, rayleigh: np.ndarray, compton: np.ndarray):
        '''call this or the from_file class method to construct the object'''
        self.energies = energies
        self.attenuations = dict()
        self.attenuations[AttenuationType.PHOTOELECTRIC_ABSORPTION] = photoelec
        self.attenuations[AttenuationType.RAYLEIGH] = rayleigh
        self.attenuations[AttenuationType.COMPTON] = compton


    def interpolate_from(self, incident_flare: FlareSpectrum): # returns AttenuationData object
        '''interpolate standard data to fit given incident_flare energy spectrum'''
        # no data except energies
        new_att = AttenuationData(incident_flare.energies, [], [], [])
        for key, att in self.attenuations.items():
            interp_atts = np.interp(incident_flare.energies, self.energies, att)
            new_att.attenuations[key] = interp_atts
        return new_att