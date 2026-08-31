# Analytical X-ray detector model

Started for the IMPRESS CubeSat to model a cerium bromide scintillator.
We've used it to model silicon and other solid-state detectors as well as other scintillators.
Always a work in progress! Kind of a mess!

## Installation
1. Clone the repo
2. `cd` into the repo directory
3. Run `pip install -e .`

## Data sources
### attenuation data from
- https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
- https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
XCOM program gets queried via HTTP request for undownloaded data

