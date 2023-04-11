#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest


###############################################################################
# Test sequencePeakValleyFilter
###############################################################################
def test_sequencePeakValleyFilter_noPoints_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data )
    
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data, keepEnds=True )


def test_sequencePeakValleyFilter_twoDimData_valueError():
    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data )
    
    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data, keepEnds=True )


def test_sequencePeakValleyFilter_onePointsKeepEnds_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data )
    
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data, keepEnds=True )


def test_sequencePeakValleyFilter_twoPointsNotKeepEnds_valueError():
    data = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequencePeakValleyFilter( data )


def test_sequencePeakValleyFilter_twoPointsKeepEnds_pointskept():
    data = [ -0.5, 1.0 ]
    # Keep ends
    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakValleyFilter_threePointsSimple_pass():
    data = [ -0.5, 1.0, 0.0 ]

    # case 1: do not keep ends
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: keep ends
    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakValleyFilter_threePointsWithSameValue_depends():
    # case 1: last two the same
    data = [ -0.5, 1.0, 1.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: first two the same
    data = [ 1.0, 1.0, 2.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: three values the same
    data = [ 1.0, 1.0, 1.0 ]
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ 1.0, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakValleyFilter_fourPointsWithSameValue_depends():
    # case 1: last three the same
    data = [ -0.5, 1.0, 1.0, 1.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: first three the same
    data = [ 1.0, 1.0, 1.0, 2.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: first two the same, last two the same
    data = [ 1.0, 1.0, 2.0, 2.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakValleyFilter_fivePointsWithSameValue_depends():
    # case 1: three in the middle the same
    data = [ -0.5, 1.0, 1.0, 1.0, 0.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: last three the same
    data = [ -0.5, 1.0, 2.0, 2.0, 2.0 ]
    
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [  ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakValleyFilter_normalUseOnlyPeakAndValleys_pass():
    data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [ 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # Keep ends
    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [ 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # Keep ends
    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [ -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )
    # Keep ends
    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequencePeakValleyFilter_normalUseExtraPointsInSequence_pass():
    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    # Do not keep ends
    calRst = utils.sequencePeakValleyFilter( data )
    expectedRst = [ 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    # Keep ends
    calRst = utils.sequencePeakValleyFilter( data, keepEnds=True )
    expectedRst = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )
    

###############################################################################
# Test sequenceHysteresisFilter
###############################################################################
def test_sequenceHysteresisFilter_lessPoints_valueError():
    gateSize = 2.0
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )
    
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )


def test_sequenceHysteresisFilter_twoDimData_valueError():
    gateSize = 2.0
    data = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )

    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )


def test_sequenceHysteresisFilter_incorrectGateSize_valueError():
    data = [ 1.0, 2.0, 1.0 ]

    # case 1: gate size is not a scalar
    gateSize = [ ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )
    gateSize = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )
    
    # case 2: gate size not positive
    gateSize = -1.0
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )
    gateSize = 0.0
    with pytest.raises( ValueError ):
        _ = utils.sequenceHysteresisFilter( data, gateSize )


def test_sequenceHysteresisFilter_noMediumPoints_keepLast():
    # case 1: last value not within gate
    data = [ 2, 5, 4, 6, 3, 4, 1 ]
    gateSize = 3.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 5, 6, 3, 1 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: last value within gate
    data = [ 2, 5, 4, 6, 3, 4, 2 ]
    gateSize = 3.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 5, 6, 3, 2 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequenceHysteresisFilter_withMediumPoints_keepLast():
    # case 1: last value not within gate
    data = [ 2, 3, 5, 4, 5, 6, 4, 3, 4, 2, 1 ]
    gateSize = 3.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 5, 6, 3, 2, 1 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: last value within gate
    data = [ 2, 3, 5, 4, 5, 6, 4, 3, 4, 5 ]
    gateSize = 3.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 5, 6, 3, 5 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequenceHysteresisFilter_morePoints_keepLast():
    # case 1: small gate size
    data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 3, 1, 5, 3, 6, 3, 6, 4, 5, 2 ]
    gateSize = 3.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 5, 6, 2, 1, 6, 1, 5, 6, 3, 6, 2 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: larger gate size
    gateSize = 4.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 6, 2, 1, 6, 1, 5, 6, 2 ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: last point within gate
    data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 3, 1, 5, 3, 6, 3, 6, 4, 5, 4 ]
    gateSize = 3.0
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ 2, 5, 6, 2, 1, 6, 1, 5, 6, 3, 6, 4 ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_sequenceHysteresisFilter_floatPoints_keepPeakValleys():
    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    gateSize = 2
    calRst = utils.sequenceHysteresisFilter( data, gateSize )
    expectedRst = [ -0.5, -1.0, 1.5, -1.0, 1.5, 4.5, 1.0, -1.0, 3.0, 1.5, -1.5, 
                    0.5, 1.0 ]
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
