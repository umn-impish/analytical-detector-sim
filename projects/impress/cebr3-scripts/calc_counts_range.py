import os
import sys
import numpy as np
from adetsim.hafx_src.HafxSimulationContainer import HafxSimulationContainer

"""
Computes the counts incident on the CeBr3 IMPRESS detectors given a lower and upper energy boud.
"""

ENG_LOWER_BOUND = 14
ENG_UPPER_BOUND = 300
CEBR3_AREA = 43 / 4

# Directory which contains pickled outputs
opt_dir = sys.argv[1]
filez = os.listdir(opt_dir)

portion_sizes = [13, 33, 18, 38]
base = "| ".join(f"{{:<{ps}}}" for ps in portion_sizes)
which_fmt = "{} flare incident on {} detector"

print("load containers")
att_cps_cm2_d = dict()
orig_cps_cm2_d = dict()
thickness_dict = dict()
containers = [
    HafxSimulationContainer.from_file(os.path.join(opt_dir, f)) for f in filez
]
containers.sort(key=lambda c: c.goes_class)
print("done loading")

print("calculating")
for flare_con in containers:
    for cur_con in containers:
        fl = flare_con.flare_spectrum.flare
        en = cur_con.flare_spectrum.energy_edges
        att_fl = cur_con.matrices[cur_con.KDISPERSED_RESPONSE] @ fl

        restrict = np.logical_and(en >= ENG_LOWER_BOUND, en <= ENG_UPPER_BOUND)

        # value * bin width
        att_counts_cm2_sec = np.sum((att_fl * np.diff(en))[restrict[:-1]])
        orig_counts_cm2_sec = np.sum((fl * np.diff(en))[restrict[:-1]])

        fl_sz = flare_con.goes_class
        att_id = cur_con.goes_class
        ident = which_fmt.format(fl_sz, att_id)
        att_cps_cm2_d[ident] = att_counts_cm2_sec
        orig_cps_cm2_d[ident] = orig_counts_cm2_sec
        thickness_dict[ident] = cur_con.al_thick * 1e4
print("done calculating")

line_fmt = "{{:<{:d}}}| {{:<{:d}}}| {{:<{:d}.2f}}| {{:<{:d}.0f}}".format(*portion_sizes)
sep_string = "=" * ((len(portion_sizes) - 1) * 2 + sum(portion_sizes))
print(
    base.format(
        "Which",
        "Flare size",
        "Al thickness (um)",
        f"count/cm2/sec for energy in ({ENG_LOWER_BOUND}, {ENG_UPPER_BOUND}) keV",
    )
)
print(sep_string)
for i, k in enumerate(
    sorted(att_cps_cm2_d.keys(), key=lambda x: ord(x[0]) + int(x[1]))
):
    # print(line_fmt.format('original', k, thickness_dict[k], orig_cps_cm2_d[k]))
    print(line_fmt.format("attenuated", k, thickness_dict[k], att_cps_cm2_d[k]))
    if (i + 1) % 4 == 0:
        print(sep_string)

print()
print(
    base.format(
        "Which",
        "Flare size",
        "Al thickness (um)",
        f"count/sec for energy in ({ENG_LOWER_BOUND}, {ENG_UPPER_BOUND}) keV",
    )
)
print(sep_string)
for i, k in enumerate(
    sorted(att_cps_cm2_d.keys(), key=lambda x: ord(x[0]) + int(x[1]))
):
    # print(line_fmt.format('original', k, thickness_dict[k], orig_cps_cm2_d[k] * CEBR3_AREA))
    print(
        line_fmt.format(
            "attenuated", k, thickness_dict[k], att_cps_cm2_d[k] * CEBR3_AREA
        )
    )
    if (i + 1) % 4 == 0:
        print(sep_string)
