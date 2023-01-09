# FFPACK - Fatigue and Fracture PACKage

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dpzhuX/ffpack/python-package.yml?branch=main)
![GitHub](https://img.shields.io/github/license/dpzhuX/ffpack)
[![DOI](https://zenodo.org/badge/575208693.svg)](https://zenodo.org/badge/latestdoi/575208693)
[![Downloads](https://static.pepy.tech/badge/ffpack)](https://pepy.tech/project/ffpack)


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

* Load sequence generator
    * Random walk
        * Uniform random walk
    * Autoregressive moving average model
        * Normal autoregressive (AR) model
        * Normal moving average (MA) model
        * Normal ARMA model
        * Normal ARIMA model

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

* Random and probabilistic model
    * Metropolis-Hastings algorithm
        * Metropolis-Hastings sampler
    * Nataf algorithm
        * Nataf transformation

* Risk and reliability model
    * First order second moment
        * fosmMVAL
    * First order reliability method
        * formHLRF
        * formCOPT

* Utility methods
    * Derivative
    * All derivative
    * Central derivative weights
    * Cycle counting aggregation
    * Counting results to counting matrix
    * Fitter for SN curve
    * Sequence peak and valleys
    * Sequence degitization

## Document

You can find the latest documentation for setting up `FFPACK` at the [Read the Docs site](https://ffpack.readthedocs.io/en/latest/).
