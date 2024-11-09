import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append("..")

from adetsim.hafx_src.X123Stack import X123Stack
from adetsim.sim_src.FlareSpectrum import FlareSpectrum

plt.style.use(os.getenv("MPL_FIG_STYLE"))


def main():
    normal, thicker, spec = generate_normal_thicker_responses()
    avec = np.ones(normal.shape[0]) * (X123Stack.X123_DIAMETER / 2) ** 2 * np.pi
    norm_avec, thick_avec = normal @ avec, thicker @ avec

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.stairs(norm_avec, spec.energy_edges, label="0.5mm thick x-123", color="black")
    ax.stairs(thick_avec, spec.energy_edges, label="1mm thick x-123", color="red")
    ax.set_xlabel("energy (keV)")
    ax.set_ylabel("effective area (cm${}^2$")
    ax.set_title("X-123 effective areas (17mm${}^2$ geometric area)")

    ax.set_xlim(2, 100)
    ax.set_ylim(1e-4, 0.5)
    ax.legend()
    ax.grid(visible=True, which="both", axis="both")
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig("x123-0.5mm-vs-1mm.png", dpi=300)
    # plt.show()


def generate_normal_thicker_responses():
    normal = X123Stack(det_thick=0.05)
    thicker = X123Stack(det_thick=0.1)

    edges = np.arange(1.5, 300.1, 0.1)
    example_spectrum = FlareSpectrum.make_with_battaglia_scaling(
        goes_class="M1", energy_edges=edges
    )

    norm_resp = normal.generate_detector_response_to(
        example_spectrum, disperse_energy=False
    )

    thick_resp = thicker.generate_detector_response_to(
        example_spectrum, disperse_energy=False
    )

    return norm_resp, thick_resp, example_spectrum


if __name__ == "__main__":
    main()
