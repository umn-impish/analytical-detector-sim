'''
Generate the effective area curve for a single channel
of an IMPISH-like detector.

Each channel is read out individually. IMPISH nominally has four channels,
but this number could be updated depending on what's required.

Data is saved to an asdf file; a plot is shown to demonstrate the data
'''
import asdf
import astropy.units as u
import impish_stack
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # The area is a trapezoid
    half_side = 33 << u.mm
    geometric_area = (3/2) * half_side**2

    detector = impish_stack.generate("lyso")
    de = 0.1
    energies = np.arange(0.1, 600 + de, de) << u.keV
    absorption = detector.compute_absorption(energies)


    flat = np.ones(energies.size - 1)
    effective_area = geometric_area * absorption

    data = {
        'energy_bin_edges': energies,
        'effective_area': effective_area
    }

    af = asdf.AsdfFile(tree=data)
    af.write_to('impish_scintillator_effective_area.asdf')

    fig, ax = plt.subplots(layout='constrained')
    ax.stairs(
        effective_area.to_value(u.cm**2),
        energies.to_value(u.keV),
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