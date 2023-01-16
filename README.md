# FFPACK - Fatigue and Fracture PACKage

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dpzhuX/ffpack/python-package.yml?color=brightgreen&label=Test&logo=github&logoColor=white)
![PyPI](https://img.shields.io/pypi/v/ffpack?color=brightgreen&label=PyPI&logo=python&logoColor=white)
![GitHub](https://img.shields.io/github/license/dpzhuX/ffpack?color=brightgreen&label=License&logo=gnu&logoColor=white)
![Read the Docs](https://img.shields.io/readthedocs/ffpack?color=brigthgreen&label=Docs&logo=read%20the%20docs&logoColor=white)
[![Downloads](https://static.pepy.tech/personalized-badge/ffpack?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ffpack)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.7478424-blue.svg?logo=Buffer&logoColor=white)](https://doi.org/10.5281/zenodo.7478424)


## Purpose
`FFPACK` ( Fatigue and Fracture PACKage ) is an open-source Python library for fatigue and fracture analysis. It supports cycle counting with ASTM methods, load sequence generators, fatigue damage evaluations, etc. A lot of features are under active development. `FFPACK` is designed to help engineers analyze fatigue and fracture behavior in engineering practice.

## Installation

`FFPACK` can be installed via [PyPI](https://pypi.org/project/ffpack/):

```
pip install ffpack
```

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
        * Harmonic superposition

* Load spectra and matrices
    * Cycle counting matrix
        * ASTM simple range counting matrix
        * ASTM range pair counting matrix
        * ASTM rainflow counting matrix
        * ASTM rainflow counting matrix for repeating history
        * Johannesson min max counting matrix
        * Rychlik rainflow counting matrix
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
        * FOSM with mean value method
    * First order reliability method
        * FORM with Hasofer-Lind-Rackwitz-Fiessler method
        * FORM with constrained optimization method
    * Second order reliability method
        * SORM with Breitung method
        * SORM with Tvedt method
        * SORM with Hohenbichler and Rackwitz method

* Utility 
    * Fitter
        * Fitter for SN curve
    * Counting matrix
        * Counting results to counting matrix
    * Aggregation
        * Cycle counting aggregation
    * Sequence filters
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
