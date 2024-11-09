import numpy as np
import os
import sys
import scipy.integrate
from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer
from adetsim.hafx_src.HafxMaterialProperties import SINGLE_DET_AREA

THRESHOLD_ENERGY = 10  # keV


def pileup_fraction_estimate(count_rate, pileup_time):
    # waiting time probability density function
    def pdf(t):
        return count_rate * np.exp(-count_rate * t)

    # integrate probability density function from 0 to the pileup time
    return scipy.integrate.quad(pdf, 0, pileup_time)


optim_dir = sys.argv[1]
pileup_time = 1 * 1e-6  # microsecond
opt_files = os.listdir(optim_dir)

loaded = dict()
for f in opt_files:
    con = HafxSimulationContainer.from_saved_file(os.path.join(optim_dir, f))
    loaded[con.flare_spectrum.goes_class] = con

keyz = ("C1", "C5", "M1", "M5", "X1")
cols = (
    "Flare size",
    "Attenuator thickness (um)",
    "Pileup fraction estimate",
    "Estimate error",
)
col_str = ("{:<30}" * len(cols)).format(*cols)
print(col_str)
for k in keyz:
    try:
        cur_con = loaded[k]
    except KeyError:
        continue

    threshold_energy = THRESHOLD_ENERGY
    eng = cur_con.flare_spectrum.energy_edges
    condition = (eng >= threshold_energy)[:-1]

    disp_flare = (
        cur_con.matrices[cur_con.KDISPERSED_RESPONSE] @ cur_con.flare_spectrum.flare
    )
    count_rate = (disp_flare * np.diff(eng))[condition].sum() * SINGLE_DET_AREA

    frac, err = pileup_fraction_estimate(count_rate, pileup_time)
    err_str = f"({(err * 100):.2e})%"
    print(f"{k:<30}{cur_con.al_thick*1e4:<30.1f}{frac:<30.2%}+-{err_str:<30}")
