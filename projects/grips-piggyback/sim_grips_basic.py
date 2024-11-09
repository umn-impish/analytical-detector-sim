import lzma
import os
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import grips_stack as gs
from adetsim.sim_src import FlareSpectrum as fs

plt.style.use(os.getenv("MPL_INTERACTIVE_STYLE"))

geom_area = ((40 * 40) << u.mm**2).to(u.cm**2)


def main():
    res_fn = "grips-responses.xz"
    thks = (0, 0.1, 0.25, 0.5, 1, 2, 3.3)

    if not os.path.exists(res_fn):
        responses = build_responses(res_fn, thks)
    else:
        with lzma.open(res_fn, "rb") as f:
            responses = pickle.load(f)

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 6))
    plot_diff_resp(responses, fig, ax)
    ax.set(
        title="GaGG effective area for different-thickness PLA in front of teflon + scintillator"
    )
    ax.legend(loc="center")
    plt.show()
    # fig.savefig('gagg-effa-different-PLA-in-front.png', dpi=300)


def build_responses(out_fn: str, pla_thicks: tuple[float]):
    to_save = dict()
    for pt in pla_thicks:
        print("do", pt, "mm")
        thicks = {
            "Al": (50 << u.um).to(u.cm).value,
            "pla": (pt << u.mm).to(u.cm).value,
            "teflon": (0.5 << u.mm).to(u.cm).value,
            "gagg": (5 << u.mm).to(u.cm).value,
        }

        stk = gs.GripsStack(thicks)
        edges = np.logspace(0, np.log10(300), num=1000)
        spec = np.zeros(edges.size - 1)
        dummy = fs.FlareSpectrum("n/a", spec, spec, edges)

        res = stk.generate_detector_response_to(dummy, disperse_energy=False)
        to_save[pt] = {"res": res, "edges": edges}

    with lzma.open(out_fn, "wb") as f:
        pickle.dump(to_save, f)

    return to_save


def plot_diff_resp(dat: dict[float, dict], fig, ax):
    for thick, res in dat.items():
        edges = res["edges"]
        onez = np.ones(edges.size - 1)
        ax.stairs(res["res"] @ onez * geom_area.value, edges, label=f"{thick}mm PLA")

    ax.set(
        xlabel="energy (keV)",
        ylabel="area (cm2)",
        title="gagg-ish eff areas",
        xscale="log",
        xlim=(6, 300),
        ylim=(1, geom_area.value + 1),
    )
    ax.axhline(16, label="geometric area (16 cm2)", color="black")


if __name__ == "__main__":
    main()
