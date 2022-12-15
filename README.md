# FFPACK - Fatigue and Fracture PACKage

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dpzhuX/ffpack/python-package.yml?branch=main)
![GitHub](https://img.shields.io/github/license/dpzhuX/ffpack?color=5bc72b)

## Purpose
`FFPACK` ( Fatigue and Fracture PACKage ) is an open source Python library for fatigue and fracture analysis. It supports the load cycle counting with ASTM methods, load sequence generators, fatigue damage evaluations, etc. A lot of features are under active development. `FFPACK` was designed to help engineers analyze the fatigue and fracture behavior in civil, mechanical and aerospace field.

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
* Load spectra and matrices
    * WIP

## Document

You can find a complete docummentation for setting up `FFPACK` at the [Read the Docs site](https://ffpack.readthedocs.io/en/latest/).
