import lzma
import pickle

import sys

sys.path.append("..")

from adetsim.sim_src import FlareSpectrum
import adetsim.hafx_src.HafxMaterialProperties as hmp
from adetsim.hafx_src import X123Stack, X123CdTeStack

x123_basename = f"../responses-and-areas/{{}}-x123-{{}}-resp-{{:.4e}}cm.lzma"


def build_x123_data(goes_class, detector_thickness=0.1, material=hmp.SI):
    assert material == hmp.SI or material == hmp.CDTE

    X123CorrectStack = (
        X123CdTeStack.X123CdTeStack if (material != hmp.SI) else X123Stack.X123Stack
    )

    print("build x123 data for", goes_class)
    dat = dict()
    save_fn = x123_basename.format(goes_class, material, detector_thickness)
    fs = FlareSpectrum.FlareSpectrum.make_with_battaglia_scaling(
        goes_class=goes_class, start_energy=1.1, end_energy=400, de=0.1
    )
    dat["energies"] = fs.energies
    dat["flare"] = fs.flare
    print("done flare spectrum")

    xs = X123CorrectStack(det_thick=detector_thickness)
    dat["area"] = xs.area
    dat["disp_resp"] = (
        disp_resp := xs.generate_detector_response_to(fs, disperse_energy=True)
    )
    print("done dispersed")
    dat["undisp_resp"] = (
        undisp_resp := xs.generate_detector_response_to(fs, disperse_energy=False)
    )
    print("done undispersed")

    dat["disp"] = disp_resp @ fs.flare
    print("done disp multiply")
    dat["undisp"] = undisp_resp @ fs.flare
    print("done undisp multiply")

    with lzma.open(save_fn, "wb") as f:
        pickle.dump(dat, f)
    print("done pickle")

    return dat
