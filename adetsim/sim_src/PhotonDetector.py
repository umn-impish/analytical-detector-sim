import numpy as np
from .FlareSpectrum import FlareSpectrum


class PhotonDetector:
    def generate_energy_resolution_given(
        self, incident_spectrum: FlareSpectrum
    ) -> np.ndarray:
        raise NotImplementedError(
            "Subclass PhotonDetector to implement detector-specific behavior."
        )
