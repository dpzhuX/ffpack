#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
from scipy import stats
import pytest
from unittest.mock import patch
from ffpack.rpm import NatafTransformation, AuModifiedMHSampler


###############################################################################
# Test subsetSimulation
###############################################################################
def test_subsetSimulation_dimLessOneCase_valueError( ):
    dim = 0

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )


def test_subsetSimulation_distObjsDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    distObjs = [ X1 ]
    corrMat = np.eye( dim )
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )


def test_subsetSimulation_corrMatDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0 ], [ 1.0 ] ]
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )
    

def test_subsetSimulation_corrMatNotTwoDimCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, 2.0 ]
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )


def test_subsetSimulation_corrMatNotSymmCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.5 ], [ -0.5, 1.0 ] ]
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )


def test_subsetSimulation_corrMatNotPositiveDefiniteCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -1.2 ], [ -1.2, 1.0 ] ]
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )


def test_subsetSimulation_corrMatDiagNotOneCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 2.0 ] ]
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                                           maxSubsets, probLevel )


@patch.object( AuModifiedMHSampler, "getSample" )
@patch( "numpy.random.normal" )
@patch.object( NatafTransformation, "getX" )
def test_subsetSimulation_normalCase_scalar( mock_getX, mock_normal, 
                                             mock_getSample ):
    '''
    In this example, we simulate the entire subset simulation with 2 subsets.
    The NatafTransformation is patched so that the xCoord is the same as uCoord.
    The AuModifiedMHSampler is patched so that a consistent sample will be returned.

    The first subset is the curde Monte Carlo simulation in which 20 U samples will
    be generated:
        [ [ 1.38384717,  0.76038508],
          [ 1.57822555,  0.10749794],
          [ 1.42550989, -0.66475512],
          [-0.4230153 ,  1.06448209],
          [-1.26405266,  1.52790535],
          [-0.28564551,  0.53836748],
          [-0.10069672,  0.30379318],
          [-0.70140242,  0.84908785],
          [-1.72596243,  1.58509537],
          [ 0.81050091, -1.04477837],
          [-0.005778  , -0.46747897],
          [-0.97071094,  0.47055962],
          [ 0.98501786, -1.70046527],
          [ 1.11347211, -1.92116972],
          [ 0.13429659, -1.10685547],
          [-2.08389663,  0.93778171],
          [-2.9033676 ,  1.61689037],
          [-0.76404783, -0.77518851],
          [-0.68922937, -0.85275686],
          [-1.10014381, -1.31564409] ]

    The corrosponding lsfValues from lsf function are:
        [ 1.48379884, 1.80801349, 2.46206514, 2.54641448, 2.81342797,
          2.82129858, 2.85638912, 2.89557063, 3.09960805, 3.16565918,
          3.33464321, 3.35366039, 3.50589772, 3.57112846, 3.68770298,
          3.81042563, 3.90967677, 4.08840445, 4.09034892, 4.70822001 ]
    
    Note that X samples are the same as the U samples and the order of the U samples 
    makes lsfValues already in ascending order.

    The second subset will choose 10% U samples in the first subset for MCMC. The 
    lsfLevel is 1.80801349.

    We mock the MCMC function and return a consistent sample:
        [ 0.12386626, 0.74278539 ]
    Thus, the U samples after the second round should be:
        [ [ 1.38384717,  0.76038508],
          [ 1.57822555,  0.10749794],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ],
          [ 0.12386626,  0.74278539 ] ]
    The corrosponding lsfValues from lsf function are:
        [ 1.48379884, 1.80801349, 2.38718474, 2.38718474, 2.38718474,
          2.38718474, 2.38718474, 2.38718474, 2.38718474, 2.38718474,
          2.38718474, 2.38718474, 2.38718474, 2.38718474, 2.38718474,
          2.38718474, 2.38718474, 2.38718474, 2.38718474, 2.38718474 ]
    '''
    dim = 2

    def g( X ):
        return -np.sum( X ) / np.sqrt( dim ) + 3.0

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    numSamples, maxSubsets = 20, 2
    probLevel = 0.1

    def sideEffect( *args, **kwargs ):
        return ( args[ 0 ], np.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] ) )
    mock_getX.side_effect = sideEffect
    mock_normal.return_value = ( [ [ 1.38384717, 0.76038508 ],
                                   [ 1.57822555, 0.10749794 ],
                                   [ 1.42550989, -0.66475512 ],
                                   [ -0.4230153, 1.06448209 ],
                                   [ -1.26405266, 1.52790535 ],
                                   [ -0.28564551, 0.53836748 ],
                                   [ -0.10069672, 0.30379318 ],
                                   [ -0.70140242, 0.84908785 ],
                                   [ -1.72596243, 1.58509537 ],
                                   [  0.81050091, -1.04477837 ],
                                   [ -0.005778, -0.46747897 ],
                                   [ -0.97071094, 0.47055962 ],
                                   [ 0.98501786, -1.70046527 ],
                                   [ 1.11347211, -1.92116972 ],
                                   [ 0.13429659, -1.10685547 ],
                                   [ -2.08389663, 0.93778171 ],
                                   [ -2.9033676, 1.61689037 ],
                                   [ -0.76404783, -0.77518851 ],
                                   [ -0.68922937, -0.85275686 ],
                                   [ -1.10014381, -1.31564409 ] ] )
    mock_getSample.return_value = np.array( [ 0.12386626, 0.74278539 ] )
    calPf, calAllLsfValues, calAllUSamples, calAllXSamples = rrm.\
        subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                          maxSubsets, probLevel )
    expectedPf = probLevel ** maxSubsets
    expectedAllLsfValues = \
        [ [ 1.48379884, 1.80801349, 2.46206514, 2.54641448, 2.81342797,
            2.82129858, 2.85638912, 2.89557063, 3.09960805, 3.16565918,
            3.33464321, 3.35366039, 3.50589772, 3.57112846, 3.68770298,
            3.81042563, 3.90967677, 4.08840445, 4.09034892, 4.70822001 ], 
          [ 1.48379884, 1.80801349, 2.38718474, 2.38718474, 2.38718474,
            2.38718474, 2.38718474, 2.38718474, 2.38718474, 2.38718474,
            2.38718474, 2.38718474, 2.38718474, 2.38718474, 2.38718474,
            2.38718474, 2.38718474, 2.38718474, 2.38718474, 2.38718474] ]
    expectedAllUSamples = \
        [ [ [ 1.38384717, 0.76038508 ],
            [ 1.57822555, 0.10749794 ],
            [ 1.42550989, -0.66475512 ],
            [ -0.4230153, 1.06448209 ],
            [ -1.26405266, 1.52790535 ],
            [ -0.28564551, 0.53836748 ],
            [ -0.10069672, 0.30379318 ],
            [ -0.70140242, 0.84908785 ],
            [ -1.72596243, 1.58509537 ],
            [  0.81050091, -1.04477837 ],
            [ -0.005778, -0.46747897 ],
            [ -0.97071094, 0.47055962 ],
            [ 0.98501786, -1.70046527 ],
            [ 1.11347211, -1.92116972 ],
            [ 0.13429659, -1.10685547 ],
            [ -2.08389663, 0.93778171 ],
            [ -2.9033676, 1.61689037 ],
            [ -0.76404783, -0.77518851 ],
            [ -0.68922937, -0.85275686 ],
            [ -1.10014381, -1.31564409 ]], 
          [ [ 1.38384717, 0.76038508 ],
            [ 1.57822555, 0.10749794 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ],
            [ 0.12386626, 0.74278539 ] ] ]
    expectedAllXSamples = expectedAllUSamples
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )
    np.testing.assert_allclose( expectedAllLsfValues, calAllLsfValues, atol=1e-6 )
    np.testing.assert_allclose( expectedAllUSamples, calAllUSamples, atol=1e-6 )
    np.testing.assert_allclose( expectedAllXSamples, calAllXSamples, atol=1e-6 )
