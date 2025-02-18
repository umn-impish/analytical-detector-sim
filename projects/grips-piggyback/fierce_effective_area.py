'''
Generate the effective area curve for a single channel
of an IMPISH-like detector.

Each channel is read out individually. IMPISH nominally has four channels,
but this number could be updated depending on what's required.

Data is saved to an asdf file; a plot is shown to demonstrate the data
'''
import asdf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import grips_stack as gs
from adetsim.sim_src import FlareSpectrum as fs
from adetsim.sim_src import AttenuationData as ad

if __name__ == '__main__':
    # The area is a trapezoid
    half_side = 33 << u.mm
    geometric_area = (3/2) * half_side**2
    thickness = 4 << u.mm

    # The detector expects a "diameter" argument,
    # but we're just going to ignore it

    material_stack = dict(
        # Thin Al layer to block optical light
        Al=(1 << u.um),
        # PLA with 10% infill
        pla=(1 * 0.1 << u.mm),
        lyso=thickness
    )

    # Code expects everything to be in cm
    material_stack = {
        k: v.to_value(u.cm)
        for (k, v) in material_stack.items()
    }

    detector = gs.GripsStack(material_stack)
    de = 0.1
    energies = np.arange(0.1, 600 + de, de)
    nothing = np.zeros(energies.size - 1)
    fake_spectrum = fs.FlareSpectrum(
        "none", thermal=nothing, nonthermal=nothing,
        energy_edges=energies
    )

    srm = detector.generate_detector_response_to(
        fake_spectrum,
        disperse_energy=False,
        chosen_attenuations=[ad.AttenuationType.PHOTOELECTRIC_ABSORPTION]
    )

    flat = np.ones_like(nothing)
    effective_area = (
        geometric_area.to(u.cm**2) * 
        (srm @ flat)
    )

    data = {
        'energy_bin_edges': energies,
        'effective_area': effective_area
    }

    af = asdf.AsdfFile(tree=data)
    af.write_to('impish_scintillator_effective_area.asdf')

    fig, ax = plt.subplots(layout='constrained')
    ax.stairs(
        effective_area,
        energies,
        color='black',
    )
    ax.set(
        xscale='log',
        yscale='log',
        ylim=(1, 20),
        xlim=(1, 700),
        xlabel='Energy [keV]',
        ylabel='Effective area [cm2]',
        title='IMPISH single-channel effective area (photoabsorption)'
    )
    plt.show(block=True)
    fig.savefig('impish-effective-area.png', dpi=300)