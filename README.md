# FFPACK - Fatigue and Fracture PACKage

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dpzhuX/ffpack/python-package.yml?color=brightgreen&label=Test&logo=github&logoColor=white)
![PyPI](https://img.shields.io/pypi/v/ffpack?color=brightgreen&label=PyPI&logo=python&logoColor=white)
![GitHub](https://img.shields.io/github/license/dpzhuX/ffpack?color=brightgreen&label=License&logo=gnu&logoColor=white)
![Read the Docs](https://img.shields.io/readthedocs/ffpack?color=brigthgreen&label=Docs&logo=read%20the%20docs&logoColor=white)
[![Downloads](https://static.pepy.tech/personalized-badge/ffpack?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ffpack)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.7478424-blue.svg?logo=Buffer&logoColor=white)](https://doi.org/10.5281/zenodo.7478424)


## Purpose
`FFPACK` ( Fatigue and Fracture PACKage ) is an open-source Python library for fatigue and fracture analysis. It supports ASTM cycle counting, load sequence generation, fatigue damage evaluation, etc. A lot of features are under active development. `FFPACK` is designed to help engineers analyze fatigue and fracture behavior in engineering practice.

## Installation

`FFPACK` can be installed via [PyPI](https://pypi.org/project/ffpack/):

```bash
pip install ffpack
```

## Usage

The following example shows the usage of ASTM rainflow counting,

```python
# Import the ASTM rainflow counting function
from ffpack.lcc import astmRainflowCounting

# Prepare the data
data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]

# Get counting results
results = astmRainflowCounting( data )
```

See the package document for more details and examples.

## Status

`FFPACK` is currently under active development. 

## Contents

* Fatigue damage rule
    * Palmgren-miner damage rule
        * Naive Palmgren-miner damage rule
        * Classic Palmgren-miner damage rule

* Load correction and counting
    * ASTM counting
        * ASTM level crossing counting
        * ASTM peak counting
        * ASTM simple range counting
        * ASTM range pair counting
        * ASTM rainflow counting
        * ASTM rainflow counting for repeating history
    * Johannesson counting
        * Johannesson min max counting
    * Rychlik counting
        * Rychlik rainflow counting
    * Four point counting
        * Four point rainflow counting

* Load sequence generator
    * Random walk
        * Uniform random walk
    * Autoregressive moving average model
        * Normal autoregressive (AR) model
        * Normal moving average (MA) model
        * Normal ARMA model
        * Normal ARIMA model
    * Sequence from spectrum
        * Spectral representation

* Load spectra and matrices
    * Cycle counting matrix
        * ASTM simple range counting matrix
        * ASTM range pair counting matrix
        * ASTM rainflow counting matrix
        * ASTM rainflow counting matrix for repeating history
        * Johannesson min max counting matrix
        * Rychlik rainflow counting matrix
        * Four point rainflow counting matrix
    * Wave spectra
        * Jonswap spectrum
        * Pierson Moskowitz spectrum
        * ISSC spectrum
        * Gaussian Swell spectrum
        * Ochi-Hubble spectrum
    * Wind spectra
        * Davenport spectrum with drag coefficient
        * Davenport spectrum with roughness length
        * EC1 spectrum
        * IEC spectrum
        * API spectrum
    * Sequence spectra
        * Periodogram spectrum
        * Welch spectrum

* Random and probabilistic model
    * Metropolis-Hastings algorithm
        * Metropolis-Hastings sampler
        * Au modified Metropolis-Hastings sampler
    * Nataf algorithm
        * Nataf transformation

* Risk and reliability model
    * First order second moment
        * Mean value FOSM
    * First order reliability method
        * Hasofer-Lind-Rackwitz-Fiessler FORM
        * Constrained optimization FORM
    * Second order reliability method
        * Breitung SORM
        * Tvedt SORM
        * Hohenbichler and Rackwitz SORM
    * Simulation based reliability method
        * Subset simulation

* Utility 
    * Aggregation
        * Cycle counting aggregation
    * Counting matrix
        * Counting results to counting matrix
    * Fitter
        * Fitter for SN curve
    * Sequence filter
        * Sequence peakValley filter
        * Sequence hysteresis filter
    * Degitization
        * Sequence degitization
    * Derivatives
        * Derivative
        * Central derivative weights
        * Gradient
        * Hessian matrix

## Document

You can find the latest documentation for setting up `FFPACK` at the [Read the Docs site](https://ffpack.readthedocs.io/en/latest/).

## Credits

This project was made possible by the help from [DM2L lab](https://dm2l.uconn.edu/).

## License

[GPLv3](https://github.com/dpzhuX/ffpack/blob/main/LICENSE)