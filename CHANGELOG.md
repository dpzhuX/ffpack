
# Change Log
All notable changes to this project will be documented in this file.
 
## [ 0.4.0 ] - Unreleased
 
### Added

### Changed
 
### Fixed
 
## [ 0.3.1 ] - 2023-01-26
 
### Added

- (lcc) Four point rainflow counting
- (lsg) Spectral representation
- (lsm) ISSC spectrum
- (lsm) Gaussian Swell spectrum
- (lsm) Ochi-Hubble spectrum
- (lsm) Davenport spectrum with drag coefficient
- (lsm) Davenport spectrum with roughness length
- (lsm) EC1 spectrum
- (lsm) IEC spectrum
- (lsm) API spectrum
- (lsm) Periodogram spectrum
- (lsm) Welch spectrum
- (rpm) Au modified Metropolis-Hastings sampler
- (rrm) Breitung SORM
- (rrm) Tvedt SORM
- (rrm) Hohenbichler and Rackwitz SORM
- (rrm) Subset simulation
- (utils) Sequence hysteresis filter
- (utils) Central difference weights
- (utils) Derivative
- (utils) Gradient
- (utils) Hessian matrix
- (utils) Gram-Schmidt orthogonization

### Changed

- (rrm) `fosm` was changed to `mvalFOSM`
- (rrm) `formHLRF` was changed to `hlrfFORM`
- (rrm) `formCOPT` was changed to `coptFORM`
- (fdr)  fatigue damage rule was changed to (fdm) fatigue damage model 
- (utils) `sequencePeakAndValleys` was changed to `sequencePeakValleyFilter`
- (utils) `FitterForSnCurve` was changed to `SnCurveFitter`
 
## [ 0.3.0 ] - 2023-01-08
 
### Added

- (lcc) ASTM range pair counting
- (lcc) ASTM rainflow counting for repeating history
- (lcc) Johannesson min max counting
- (lsg) Normal moving average (MA) model
- (lsg) Normal ARMA model
- (lsg) Normal ARIMA model
- (lsm) ASTM simple range counting matrix
- (lsm) ASTM range pair counting matrix
- (lsm) ASTM rainflow counting matrix
- (lsm) ASTM rainflow counting matrix for repeating history
- (lsm) Johannesson min max counting matrix
- (lsm) Rychlik rainflow counting matrix
- (lsm) Pierson Moskowitz spectrum
- (lsm) Jonswap spectrum
- (rpm) Metropolis-Hastings sampler
- (rpm) Nataf transformation
- (rrm) fosm
- (rrm) formHLRF
- (rrm) formCOPT
- (utils) Counting results to counting matrix

## [ 0.2.0 ] - 2022-12-23
 
### Added

- (fdr) Palmgren-miner damage rule
- (lcc) Rychlik counting method
- (lsg) Autoregressive model
- (lsg) Random walk
- (utils) Cycle counting aggregation
- (utils) Fitter for SN curve

### Changed

- (utils) `getSequencePeakAndValleys` was changed to `sequencePeakAndValleys`
 
## [ 0.1.0 ] - 2022-12-11
 
### Added

- ASTM load cycle counting methods
- Utility methods to digitize sequence data
