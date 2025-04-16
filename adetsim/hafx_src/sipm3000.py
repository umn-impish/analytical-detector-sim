from .detectors import LinearInterpolateDisperseDetector

# XXX: the photon detector isn't the main constraint on energy resolution. the scintillator does.
#      => move photon detector stuff to scintillator.


class Sipm3000(LinearInterpolateDisperseDetector):
    # guess for now
    DE_20KEV = 0.4
    DE_667KEV = 0.045

    def __init__(self):
        super().__init__(20, 667, Sipm3000.DE_20KEV, Sipm3000.DE_667KEV)
