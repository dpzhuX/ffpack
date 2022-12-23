# FFPACK - Fatigue and Fracture PACKage

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dpzhuX/ffpack/python-package.yml?branch=main)
![GitHub](https://img.shields.io/github/license/dpzhuX/ffpack)

## Purpose
`FFPACK` ( Fatigue and Fracture PACKage ) is an open-source Python library for fatigue and fracture analysis. It supports load cycle counting with ASTM methods, load sequence generators, fatigue damage evaluations, etc. A lot of features are under active development. `FFPACK` is designed to help engineers analyze fatigue and fracture behavior in engineering practice.

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

* Load cycle counting
    * ASTM
        * ASTM level crossing counting
        * ASTM peak counting
        * ASTM simple range counting
        * ASTM rainflow counting
    * Rychlik
        * Rychlik rainflow Counting

* Load sequence generator
    * Random walk
        * Uniform random walk
    * Autoregressive model
        * Normal autoregressive model

* Utility methods
    * Cycle counting aggregation
    * Fitter for SN curve
    * Sequence peak and valleys
    * Sequence degitization

## Document

You can find a complete documentation for setting up `FFPACK` at the [Read the Docs site](https://ffpack.readthedocs.io/en/latest/).
