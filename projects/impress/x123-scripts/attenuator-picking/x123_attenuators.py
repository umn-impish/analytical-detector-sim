import astropy.units as u
import numpy as np
import pickle

from adetsim.sim_src import FlareSpectrum
from adetsim.sim_src.Material import Material
from adetsim.sim_src.AttenuationData import AttenuationData

import adetsim.hafx_src.HafxMaterialProperties as hmp
from adetsim.hafx_src import X123Stack


@u.quantity_input
def attenuate(
    pinhole_diam: u.cm, thicks: u.um, attenuator_mat: Material, energy_edges: u.keV
):
    # pinhole through the center
    pinhole_area = (np.pi * (diam / 2) ** 2).to(u.mm**2)

    # attenuated area excluding the pinhole
    x123_area = np.pi * (X123Stack.X123Stack.X123_DIAMETER) ** 2 << u.mm**2
    x123_eff_area = x123_area - pinhole_area

    classes = ("C1", "M1", "M5", "X1")
    flares = [
        FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(
            goes_class=gc, energy_edges=energy_edges.to(u.keV).value
        )
        for gc in classes
    ]

    unatt_xs = X123Stack.X123Stack()
    unatt_res = unatt_xs.generate_detector_response_to(flares[0], disperse_energy=True)
    unatt_res *= (pinhole_area / x123_area).to(u.one).value

    processed = {cl: dict() for cl in classes}

    thicks = thicks.to(u.um)
    for thk in thicks:
        print(f"start {thk} thick {attenuator_mat.name} attenuator")
        xs = X123Stack.X123Stack()
        attenuator_mat.diameter = xs.materials[0].diameter
        attenuator_mat.thickness = thk.to(u.cm).value

        xs.materials.insert(0, attenuator)
        att_res = xs.generate_detector_response_to(flares[0], disperse_energy=True)
        att_res *= (x123_eff_area / x123_area).to(u.one).value

        res = att_res + unatt_res
        for fl in flares:
            print("\tdo", fl.goes_class)
            processed[fl.goes_class][f"{thk:.2f}"] = res @ fl.flare

    return processed


if __name__ == "__main__":
    # al
    # thicks = (np.array([100, 105, 110, 115, 120]) << u.um).to(u.cm)
    # w
    # thicks = np.array([1, 2, 3, 4, 5]) << u.micron
    # au
    thicks = np.array([1, 3, 5]) << u.micron
    # ti
    # thicks = np.array([5, 10, 20]) << u.micron
    edges = np.arange(1.1, 20, 0.02) << u.keV
    diam = 50 << u.micron

    mat_key = hmp.AU
    attenuator = Material(
        diameter=NotImplemented,
        attenuation_thickness=NotImplemented,
        mass_density=hmp.DENSITIES[mat_key],
        attenuation_data=AttenuationData.from_compound_dict(
            hmp.ATTEN_FORMULAS[mat_key]
        ),
        name=mat_key,
    )

    dat = attenuate(diam, thicks, attenuator, edges)

    print(dat)
    with open("x123-atts.pkl", "wb") as f:
        pickle.dump({"dat": dat, "diameter": diam, "edges": edges}, f)
