#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest


###############################################################################
# Test sequencePeakAndValleys
###############################################################################
def test_sequencePeakAndValleys_noPoints_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data )
    
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data, keepEnds=True )


def test_sequencePeakAndValleys_twoDimData_valueError():
    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data )
    
    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data, keepEnds=True )


def test_sequencePeakAndValleys_onePointsKeepEnds_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data )
    
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data, keepEnds=True )


def test_sequencePeakAndValleys_twoPointsNotKeepEnds_valueError():
    data = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakAndValleys( data )


def test_sequencePeakAndValleys_twoPointsKeepEnds_pointskept():
    data = [ -0.5, 1.0 ]
    # Keep ends
    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakAndValleys_threePointsSimple_pass():
    data = [ -0.5, 1.0, 0.0 ]

    # case 1: do not keep ends
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: keep ends
    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakAndValleys_threePointsWithSameValue_depends():
    # case 1: last two the same
    data = [ -0.5, 1.0, 1.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: first two the same
    data = [ 1.0, 1.0, 2.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: three values the same
    data = [ 1.0, 1.0, 1.0 ]
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ 1.0, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakAndValleys_fourPointsWithSameValue_depends():
    # case 1: last three the same
    data = [ -0.5, 1.0, 1.0, 1.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: first three the same
    data = [ 1.0, 1.0, 1.0, 2.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: first two the same, last two the same
    data = [ 1.0, 1.0, 2.0, 2.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakAndValleys_fivePointsWithSameValue_depends():
    # case 1: three in the middle the same
    data = [ -0.5, 1.0, 1.0, 1.0, 0.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: last three the same
    data = [ -0.5, 1.0, 2.0, 2.0, 2.0 ]
    
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakAndValleys_normalUseOnlyPeakAndValleys_pass():
    data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [ 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    # Keep ends
    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [ 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    # Keep ends
    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [ -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    # Keep ends
    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakAndValleys_normalUseExtraPointsInSequence_pass():
    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakAndValleys( data )
    expectedRst = [ 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    # Keep ends
    calRst = utils.sequencePeakAndValleys( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )
    


###############################################################################
# Test digitizeSequenceToResolution
###############################################################################
def test_digitizeSequenceToResolution_normalUseCase_pass():
    data = [ 1.2, 2.6, 3.4, 0.8 ]
    calRst = utils.sequenceDigitization( data, resolution=0.5 )
    expectedRst = [ 1.0, 2.5, 3.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.2, -2.6, -3.4, -0.8 ]
    calRst = utils.sequenceDigitization( data, resolution=0.5 )
    expectedRst = [ -1.0, -2.5, -3.5, -1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    calRst = utils.sequenceDigitization( data, resolution=0.1 )
    expectedRst = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    calRst = utils.sequenceDigitization( data, resolution=0.5 )
    expectedRst = [ -1.0, 2.5, 2.0, 0.5, -0.5, 1.0, -1.5, -2.5, 3.5, 0.5, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    calRst = utils.sequenceDigitization( data, resolution=1.0 )
    expectedRst = [ -1.0, 2.0, 2.0, 1.0, 0.0, 1.0, -2.0, -2.0, 3.0, 0.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -2.5, -1.5, -0.5, 0.5, 1.5, 2.5 ]
    calRst = utils.sequenceDigitization( data, resolution=1.0 )
    expectedRst = [ -2.0, -2.0, 0.0, 0.0, 2.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_digitizeSequenceToResolution_twoPointsCase_pass():
    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=2.0 )
    expectedRst = [ 4.0, -4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=1.5 )
    expectedRst = [ 3.0, -3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=1.0 )
    expectedRst = [ 3.0, -3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=0.5 )
    expectedRst = [ 3.0, -3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=0.2 )
    expectedRst = [ 3.2, -3.2 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=0.1 )
    expectedRst = [ 3.1, -3.1 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=0.05 )
    expectedRst = [ 3.15, -3.15 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=0.01 )
    expectedRst = [ 3.14, -3.14 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.sequenceDigitization( data, resolution=0.001 )
    expectedRst = [ 3.142, -3.142 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_digitizeSequenceToResolution_emptyInputCase_pass():
    data = [ ]
    calRst = utils.sequenceDigitization( data, resolution=0.001 )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_digitizeSequenceToResolution_twoDimInputCase_valueError():
    data = [ [ 1.0, 2.5 ], [ 3.0, 4.5 ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceDigitization( data, resolution=1.0 )



###############################################################################
# Test centralDiffWeights
###############################################################################
def test_centralDiffWeights_npLeNdivPlusOne_valueError():
    Np = 5
    ndiv = 5
    with pytest.raises( ValueError ):
        _ = utils.centralDiffWeights( Np, ndiv )

    ndiv = 6
    with pytest.raises( ValueError ):
        _ = utils.centralDiffWeights( Np, ndiv )


def test_centralDiffWeights_npIsEven_valueError():
    Np = 4
    ndiv = 1
    with pytest.raises( ValueError ):
        _ = utils.centralDiffWeights( Np, ndiv )


@pytest.mark.parametrize( "Np", [ 3, 5, 7, 9 ])
def test_centralDiffWeights_ndivIsOne_array( Np ):
    ndiv = 1
    expectedWeights = [ ]
    if Np == 3:
        expectedWeights = np.array( [ -1, 0, 1 ] ) / 2.0
    elif Np == 5:
        expectedWeights = np.array( [ 1, -8, 0, 8, -1 ] ) / 12.0
    elif Np == 7:
        expectedWeights = np.array( [ -1, 9, -45, 0, 45, -9, 1 ] ) / 60.0
    elif Np == 9:
        expectedWeights = np.array( [ 3, -32, 168, -672, 0, 
                                      672, -168, 32, -3 ] ) / 840.0
    
    calWeights = utils.centralDiffWeights( Np, ndiv )
    np.testing.assert_allclose( np.round( calWeights, 5 ), 
                                np.round( expectedWeights, 5 ) )


@pytest.mark.parametrize( "Np", [ 3, 5, 7, 9 ])
def test_centralDiffWeights_ndivIsTwo_array( Np ):
    ndiv = 2
    expectedWeights = [ ]
    if Np == 3:
        expectedWeights = np.array( [ 1, -2.0, 1 ] )
    elif Np == 5:
        expectedWeights = np.array( [ -1, 16, -30, 16, -1 ] ) / 12.0
    elif Np == 7:
        expectedWeights = np.array( [ 2, -27, 270, -490, 270, -27, 2 ] ) / 180.0
    elif Np == 9:
        expectedWeights = np.array( [ -9, 128, -1008, 8064, -14350, 
                                      8064, -1008, 128, -9 ] ) / 5040.0
    
    calWeights = utils.centralDiffWeights( Np, ndiv )
    np.testing.assert_allclose( np.round( calWeights, 5 ), 
                                np.round( expectedWeights, 5 ) )


###############################################################################
# Test derivative
###############################################################################
def test_derivative_orderLeNPlusOne_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.derivative( f, 1.0, dx=1e-6, n=3, order=3 )

    with pytest.raises( ValueError ):
        _ = utils.derivative( f, 1.0, dx=1e-6, n=4, order=3 )


def test_derivative_orderIsEven_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.derivative( f, 1.0, dx=1e-6, order=4 )


@pytest.mark.parametrize( "x0", np.linspace( -10, 10, 11 ) )
def test_derivative_linearFun_scalar( x0 ):
    f = lambda x: 2 * x 
    df = lambda x: 2

    calRst = utils.derivative( f, x0, dx=1e-6 )
    expectedRst = df( x0 )
    np.testing.assert_allclose( calRst, expectedRst )


@pytest.mark.parametrize( "x0", np.linspace( -10, 10, 11 ) )
def test_derivative_nonLinearFunCase1_scalar( x0 ):
    f = lambda x: x ** 3 + 2 * x ** 2 + 7 * x
    df = lambda x: 3 * x ** 2 + 4 * x + 7

    calRst = utils.derivative( f, x0, dx=1e-6 )
    expectedRst = df( x0 )
    np.testing.assert_allclose( calRst, expectedRst )


@pytest.mark.parametrize( "x0", np.linspace( -10, 10, 11 ) )
def test_derivative_nonLinearFunCase2_scalar( x0 ):
    f = lambda x: x ** 2 + 2 * np.exp( x ) + 7 * x
    df = lambda x: 2 * x + 2 * np.exp( x ) + 7

    calRst = utils.derivative( f, x0, dx=1e-6 )
    expectedRst = df( x0 )
    np.testing.assert_allclose( calRst, expectedRst )

