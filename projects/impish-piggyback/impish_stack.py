import astropy.units as u

from adetsim import detection, matter

MAT_FORMULAS = {
    "teflon": {"C": 2, "F": 4},
    "pla": {"C": 3, "H": 4, "O": 2},
    "gagg": {"Gd": 2.9995, "Ce": 0.0005, "Al": 2, "Ga": 5, "O": 12},
    "lyso": {"Lu": 1.98, "Y": 0.02, "Si": 1, "O": 5},
    "yap": {"Y": 1, "Al": 1, "O": 3},
    "delrin": {"C": 1, "H": 2, "O": 1},
}

MAT_DAT = {
    k: matter.Attenuations.from_compound_dict(d) for (k, d) in MAT_FORMULAS.items()
}
MAT_DAT["aluminum"] = matter.Attenuations.from_element("Al")

MAT_DENS = {
    "teflon": 2.2,
    "pla": 1.24,
    "lyso": 7.1,
    "gagg": 6.63,
    "yap": 5.4,
    "delrin": 1.4,
    "aluminum": 2.7,
}
MAT_DENS = {k: v << (u.g / u.cm**3) for (k, v) in MAT_DENS.items()}


def generate(scintillator: str) -> detection.PhysicalDetectorStack:
    """Generate the detector stack that corresponds to the IMPISH geometry."""
    setup = (
        ("aluminum", 100 << u.um),
        ("delrin", 3.2 << u.mm),
        ("teflon", 750 << u.um),
        (scintillator, 4 << u.mm),
    )

    materials = []
    for n, thk in setup:
        mat = matter.Material(
            name=n,
            thickness=thk,
            density=MAT_DENS[n],
            attenuations=MAT_DAT[n],
            average_across_bins=False,
        )
        mat.attenuations.compton = False
        mat.attenuations.photoelectric = mat.attenuations.rayleigh = True
        materials.append(mat)

    return detection.PhysicalDetectorStack(materials)
