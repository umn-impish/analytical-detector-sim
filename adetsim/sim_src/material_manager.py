import copy
from collections import OrderedDict
import os
import requests

import astropy.units as u
import asdf
import numpy as np

from . import material_constants as mcon
from . import magic_data

CACHE_PATH = os.path.join(os.path.dirname(__file__), "element_cache")
FILE_FMT = os.path.join(CACHE_PATH, "{elt}.asdf")


def fetch_element(element_name: str) -> dict[str, u.Quantity]:
    element_name = element_name.title()
    if element_name not in mcon.elements:
        raise ValueError(f"no data is available for {element_name} from NIST")

    file_name = download_save_nist(element_name)
    return load_element_data(file_name)


def fetch_compound(formula: dict[str, float]) -> dict[str, dict[str, u.Quantity]]:
    """
    Scale mass attenuation coefficients by compound or mixture mass.
    Still need to be interpolated properly if you wanna combine them.

    Example formula for CeBr3: {'Ce': 1, 'Br': 3}
    Example for water:         {'H': 2, 'O': 1}
    Example for PTFE:          {'C': 2, 'F': 4}

    Doping can also be handled.
    Example for GAGG(Ce):      {'Gd': 2.95, 'Ce': 0.05, 'Al': 2, 'Ga': 3, 'O': 12}
    """
    masses = dict()
    coeffs = dict()
    for element, num in formula.items():
        element = element.title()
        coeffs[element] = fetch_element(element)
        masses[element] = num * mcon.atomic_masses[mcon.elements[element]]

    total_mass = sum(masses.values())
    scaled_masses = {k: (v / total_mass) for (k, v) in masses.items()}
    ret = dict()
    for elt, m in scaled_masses.items():
        ret_key = f"{elt}_{m:0.2f}"

        # Do not scale this
        energy = coeffs[elt].pop("energy")
        ret[ret_key] = {k: (v * m) for (k, v) in coeffs[elt].items()}
        ret[ret_key]["energy"] = energy
    return ret


def download_save_nist(name: str) -> str:
    """
    Request photoelectric, incoherent, coherent scattering data from NIST XCOM program.
    Return the resulting file name
    """
    elt_file = FILE_FMT.format(elt=name)
    if os.path.exists(elt_file):
        # Have data already
        return elt_file
    print(f"{name} has no local data; downloading")
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)

    atomic_number = mcon.elements[name]

    # Data request URL from XCOM
    element_url = "https://physics.nist.gov/cgi-bin/Xcom/data.pl"
    resp = requests.post(
        element_url,
        data=magic_data.request.format(znum=atomic_number),
        headers=magic_data.headers
    )
    try:
        data = decode_nist_response(resp.text)
    except ValueError:
        raise ValueError(
            "Issue decoding NIST output. Make sure your values are OK."
            "NIST response was:\n" + resp.text
        )

    out = asdf.AsdfFile(data)
    out.write_to(elt_file)
    return elt_file


def decode_nist_response(txt: str) -> dict[str, u.Quantity]:
    """Decode the text reply from NIST into numerical arrays"""
    lines = txt.split("\n")
    data = []
    # Last line is empty, so skip it
    for d in reversed(lines[:-1]):
        if not d:
            break
        data.append([float(x) for x in d.strip().split()])

    energy, ray, comp, photo = np.array(list(reversed(data))).T
    # Some absorption edges have two energy values really close
    # Make them a little further apart for numerics
    eps = 1e-10
    for i in range(energy.size - 1):
        if np.abs(energy[i] - energy[i + 1]) < eps:
            energy[i + 1] += eps
    return {
        "energy": energy << u.MeV,
        "rayleigh": ray << u.cm**2 / u.g,
        "compton": comp << u.cm**2 / u.g,
        "photoelectric": photo << u.cm**2 / u.g,
    }


def load_element_data(fn: str) -> dict[str, u.Quantity]:
    # Keys we wanna keep from the data file
    keep = ["energy", "photoelectric", "compton", "rayleigh"]
    with asdf.open(fn) as f:
        return {k: copy.deepcopy(f[k]) for k in keep}
