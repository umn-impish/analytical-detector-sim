import os

""" a bunch of constants """
AL = "Al"
AU = "Au"
TEF = "Teflon"
BE = "Be"
CEBR3 = "CeBr3"
SI = "Si"
CDTE = "CdTe"
W = "W"
TI = "Ti"
# order on HaFX detector
HAFX_MATERIAL_ORDER = [AL, TEF, BE, CEBR3]
HAFX_DEAD_TIME = 2e-6  # s

"""
NB: these all need to get re-verified. i just took them from Ethan's code.
update 24 sep 2021: updated to match Geant sims (teflon is like 10x thicker than I had. whoops)
update 28 sep 2021: went back to old thickness; G4 thickness was ridiculous and overattenuating
"""
BE_THICKNESS = 0.075  # cm
TEFLON_THICKNESS = 0.0127  # cm
CEBR3_THICKNESS = 0.5  # cm
THICKNESSES = {
    BE: BE_THICKNESS,
    TEF: TEFLON_THICKNESS,
    CEBR3: CEBR3_THICKNESS,
    # aluminum is special so we leave it out
}

ATTEN_FORMULAS = {
    AL: {"Al": 1},
    AU: {"Au": 1},
    TEF: {"C": 2, "F": 4},
    BE: {"Be": 1},
    CEBR3: {"Ce": 1, "Br": 3},
    SI: {"Si": 1},
    CDTE: {"Cd": 1, "Te": 1},
    W: {"W": 1},
    TI: {"Ti", 1},
}

# all g/cm3
DENSITIES = {
    AL: 2.712,
    AU: 19.3,  # wowza
    W: 19.3,  # also wowza
    BE: 1.850,
    TEF: 2.200,
    CEBR3: 5.1,
    SI: 2.33,
    CDTE: 5.85,
    TI: 4.5,
}

FULL_AREA = 43  # cm2
SINGLE_DET_AREA = FULL_AREA / 4  # cm2
DIAMETER = 3.7  # cm
