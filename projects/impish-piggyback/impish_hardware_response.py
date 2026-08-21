from typing import cast

import asdf
import astropy.units as u
import impish_stack
import numpy as np

from adetsim import detection


def main():
    print("start hardware response")
    response = generate_hardware_responses()
    
    print("start lyso resolution")
    response["lyso-resolution"] = (
        generate_energy_resolution(
            energy_bins=response["energy_bins"],
            anchor_energies=[31, 60, 122] << u.keV,
            anchor_fwhms=(lyso_fwhms := [60, 30, 21] << u.percent),
            fwhm_errors=0.05 * lyso_fwhms,
        )
        << u.one
    )

    print("start yap resolution")
    response["yap-resolution"] = (
        generate_energy_resolution(
            energy_bins=response["energy_bins"],
            anchor_energies=[31, 60, 122] << u.keV,
            anchor_fwhms=(yap_fwhms := [28, 20, 13] << u.percent),
            fwhm_errors=0.05 * yap_fwhms,
        )
        << u.one
    )
    af = asdf.AsdfFile(response)
    af.write_to("impish-hardware-response.asdf")


def generate_hardware_responses() -> dict[str, u.Quantity]:
    area = {
        "lyso": cast(u.Quantity, (3 / 2) * (33 * 33 << u.mm**2)),
        # The YAP area is smaller because we needed to make a last-minute
        # design change to its dimensions.
        "yap": cast(
            u.Quantity, (33 * 29 << u.mm**2) + (1 / 2) * ((62 - 33) * 29 << u.mm**2)
        ),
    }

    # Collimation open area computed for (1 / 6) the hexagonal section
    wall_thickness = 1.02 << u.mm
    rad = 4 << u.mm
    open = (1 / 2) * rad**2 * np.sin(60 << u.deg)
    blocked = (wall_thickness / 2) ** 2 / np.tan(60 << u.deg) + (
        rad * wall_thickness
    ) / 2
    open_fraction = (open / (open + blocked)).to_value(u.one)

    de = 0.1
    energy_bins = np.arange(10, 400 + de, de) << u.keV
    output = {"energy_bins": energy_bins}
    for material in ("yap", "lyso"):
        stack = impish_stack.generate(material)
        response = np.diag(
            stack.compute_absorption(energy_edges=energy_bins).to_value(u.one)
        ) << (u.ct / u.ph)
        output[f"{material}-matrix"] = (
            open_fraction * response * area[material].to(u.cm**2)
        )

    return output


@u.quantity_input
def generate_energy_resolution(
    energy_bins: u.Quantity[u.keV],
    anchor_energies: u.Quantity[u.keV],
    anchor_fwhms: u.Quantity[u.percent],
    fwhm_errors: u.Quantity[u.percent],
) -> np.ndarray:
    """
    Generate the energy resolution component for a scintillator
    """
    res_gen = detection.SqrtEnergyResolution(
        reference_fwhms=anchor_fwhms,
        fwhm_errors=fwhm_errors,
        reference_energies=anchor_energies,
    )
    return res_gen.generate_resolution_matrix(energy_bins)


if __name__ == "__main__":
    main()
