import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from adetsim.sim_src.FlareSpectrum import FlareSpectrum

MIN_NUMBER = 156
BIN_WIDTH = 1 / 16

goes_class = "C1"
starte, ende, de = 1, 300, 0.05
edges = np.arange(starte, ende + de, de)
fs = FlareSpectrum.make_with_battaglia_scaling(
    goes_class=goes_class, energy_edges=edges
)

try:
    trev_e, trev_f = np.loadtxt(f"trevor-auto-extracted-{goes_class}.tab", unpack=True)
except OSError:
    trev_e, trev_f = np.zeros(0), np.zeros(0)


def main():
    calc_cps()
    plot_compare()


def calc_cps():
    examine = {"trevor": [trev_e, trev_f], "william": [fs.energy_edges, fs.flare]}
    cebr3_cutoff = 20

    rois = ((11, 26), (11, 300))
    for roi in rois:
        roi_start, roi_end = roi
        for n, energy_and_flare in examine.items():
            e, f = energy_and_flare

            restrict = (e >= roi_start) & (e <= roi_end)
            restrict_cebr3_cutoff = (e >= cebr3_cutoff) & (e <= roi_end)

            if e.size == f.size:
                counts_cm2_sec = np.trapz(y=f[restrict], x=e[restrict])
                counts_cm2_sec_realistic = np.trapz(
                    y=f[restrict_cebr3_cutoff], x=e[restrict_cebr3_cutoff]
                )
            else:
                counts_cm2_sec = (f * np.diff(e))[restrict[:-1]].sum()
                counts_cm2_sec_realistic = (f * np.diff(e))[
                    restrict_cebr3_cutoff[:-1]
                ].sum()

            print(n, "-->", goes_class)
            print(f"\t{roi_start}-{roi_end} keV: {counts_cm2_sec:.2f} counts/cm2/sec")
            print(
                f"\t\tRequired effective area ({BIN_WIDTH*1000:.2f} ms time bin): {MIN_NUMBER / (counts_cm2_sec * BIN_WIDTH):.2f} cm2"
            )
            print(
                f"\t{cebr3_cutoff}-{roi_end} keV: {counts_cm2_sec_realistic:.2f} counts/cm2/sec"
            )
            print(
                f"\t\tRequired effective area ({BIN_WIDTH*1000:.2f} ms time bin): {MIN_NUMBER / (counts_cm2_sec_realistic * BIN_WIDTH):.2f} cm2"
            )


def plot_compare():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(trev_e, trev_f, label="Trevor", color="orange", linewidth=3)
    ax.stairs(
        fs.flare,
        fs.energy_edges,
        label="William",
        color="blue",
        linestyle="--",
        linewidth=3,
    )
    ax.legend()

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(
        which="both", bottom=True, top=True, left=True, right=True, direction="in"
    )
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=4)

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Photon flux (photon keV${}^{-1}$ cm${}^{-2}$ s${}^{-1}$)")
    ax.set_title(f"Compare Trevor and William spectra, {goes_class} flare")

    ax.set_ylim(1e-4, 3e7)
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(f"trevor-william-compare-{goes_class}.pdf", dpi=300)


if __name__ == "__main__":
    main()
