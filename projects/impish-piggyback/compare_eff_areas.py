import astropy.units as u
import impish_stack
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("nice.mplstyle")

# Trapezoidal crystals
geom_area = 1.5 * (33 * 33) << u.mm**2


def main():
    edges = np.logspace(0, np.log10(300), num=1000) << u.keV

    srms = {}
    for scintillator in ("gagg", "lyso", "yap"):
        stack = impish_stack.generate(scintillator=scintillator)
        response = stack.compute_absorption(edges)
        srms[scintillator] = np.diag(response)

    color_map = {"gagg": "black", "lyso": "red", "yap": "blue"}
    name_map = {"gagg": "GAGG", "lyso": "LYSO", "yap": "YAP"}

    _, ax = plt.subplots(layout="constrained")
    area_vector = geom_area * np.ones(edges.size - 1)
    for name, srm in srms.items():
        eff_area = srm @ area_vector
        ax.stairs(
            eff_area.to_value(u.cm**2),
            edges.to_value(u.keV),
            color=color_map[name],
            label=name_map[name],
        )
    ax.set(
        xlabel="Energy (keV)",
        ylabel="Area (cm$^2$)",
        title="Effective area curves: LYSO, GAGG, YAP",
        xscale="log",
        yscale="log",
        xlim=(7, 300),
        ylim=(0.1, 20),
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
