#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest

###############################################################################
# Test getSequencePeakAndValleys
###############################################################################
def test_getSequencePeakAndValleys_normalUseOnlyPeakAndValleys_pass():
    data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    # Do not keep ends
    calRst = utils.getSequencePeakAndValleys( data )
    expectedRst= [ 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    # Keep ends
    calRst = utils.getSequencePeakAndValleys( data, keepEnds=True )
    expectedRst= [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    # Do not keep ends
    calRst = utils.getSequencePeakAndValleys( data )
    expectedRst= [ 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    # Keep ends
    calRst = utils.getSequencePeakAndValleys( data, keepEnds=True )
    expectedRst= [ 1.0, 2.0, 0.5, 3.0, 1.0, 4.5, 2.5, 3.5, 1.5, 4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    # Do not keep ends
    calRst = utils.getSequencePeakAndValleys( data )
    expectedRst= [ -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    # Keep ends
    calRst = utils.getSequencePeakAndValleys( data, keepEnds=True )
    expectedRst= [ -1.0, -2.0, -0.5, -3.0, -1.0, -4.5, -2.5, -3.5, -1.5, -4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_getSequencePeakAndValleys_normalUseExtraPointsInSequence_pass():
    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    # Do not keep ends
    calRst = utils.getSequencePeakAndValleys( data )
    expectedRst= [ 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -0.5, 0.0, 1.0, -1.0, -2.0, -1.0, 1.5, 3.0, 2.5, -1.0, 0.5, 1.5, 4.5, 
             3.5, 1.0, -1.0, -2.5, -1.5, 3.0, 3.5, 1.5, 0.0, -1.5, 0.5, 1.0 ]
    # Keep ends
    calRst = utils.getSequencePeakAndValleys( data, keepEnds=True )
    expectedRst= [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_getSequencePeakAndValleys_threePointsCase_pass():
    data = [ -0.5, 1.0, 0.0 ]
    # Do not keep ends
    calRst = utils.getSequencePeakAndValleys( data )
    expectedRst= [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -0.5, 1.0, 0.0 ]
    # Keep ends
    calRst = utils.getSequencePeakAndValleys( data, keepEnds=True )
    expectedRst= [ -0.5, 1.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_getSequencePeakAndValleys_twoPointsKeepEndsCase_pass():
    data = [ -0.5, 1.0 ]
    # Keep ends
    calRst = utils.getSequencePeakAndValleys( data, keepEnds=True )
    expectedRst= [ -0.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_getSequencePeakAndValleys_twoPointsNotKeepEndsCase_valueError():
    data = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data )

def test_getSequencePeakAndValleys_onePointsKeepEndsCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data )
    
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data, keepEnds=True )
    
def test_getSequencePeakAndValleys_noPointsAndTwoDimCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data )
    
    data = [ ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data, keepEnds=True )
    
    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data )
    
    data = [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.getSequencePeakAndValleys( data, keepEnds=True )


###############################################################################
# Test digitizeSequenceToResolution
###############################################################################
def test_digitizeSequenceToResolution_normalUseCase_pass():
    data = [ 1.2, 2.6, 3.4, 0.8 ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.5 )
    expectedRst = [ 1.0, 2.5, 3.5, 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.2, -2.6, -3.4, -0.8 ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.5 )
    expectedRst = [ -1.0, -2.5, -3.5, -1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.1 )
    expectedRst = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.5 )
    expectedRst = [ -1.0, 2.5, 2.0, 0.5, -0.5, 1.0, -1.5, -2.5, 3.5, 0.5, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=1.0 )
    expectedRst = [ -1.0, 2.0, 2.0, 1.0, 0.0, 1.0, -2.0, -2.0, 3.0, 0.0, 0.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ -2.5, -1.5, -0.5, 0.5, 1.5, 2.5 ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=1.0 )
    expectedRst = [ -2.0, -2.0, 0.0, 0.0, 2.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_digitizeSequenceToResolution_twoPointsCase_pass():
    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=2.0 )
    expectedRst = [ 4.0, -4.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=1.5 )
    expectedRst = [ 3.0, -3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=1.0 )
    expectedRst = [ 3.0, -3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.5 )
    expectedRst = [ 3.0, -3.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.2 )
    expectedRst = [ 3.2, -3.2 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.1 )
    expectedRst = [ 3.1, -3.1 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.05 )
    expectedRst = [ 3.15, -3.15 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.01 )
    expectedRst = [ 3.14, -3.14 ]
    np.testing.assert_allclose( calRst, expectedRst )

    data = [ np.pi, -np.pi ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.001 )
    expectedRst = [ 3.142, -3.142 ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_digitizeSequenceToResolution_emptyInputCase_pass():
    data = [ ]
    calRst = utils.digitizeSequenceToResoultion( data, resolution=0.001 )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_digitizeSequenceToResolution_twoDimInputCase_valueError():
    data = [ [ 1.0, 2.5 ], [ 3.0, 4.5 ] ]
    with pytest.raises( ValueError ):
        _ = utils.digitizeSequenceToResoultion( data, resolution=1.0 )


###############################################################################
# Test cycleCountingAccordingToBinSize
###############################################################################
def test_cycleCountingAccordingToBinSize_onePairDefaultBinSize_oneCount():
    # case 1: closer to 0
    data = [ [ 0.2, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: closer to 1
    data = [ [ 0.7, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: in the middle
    data = [ [ 0.5, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 4: greater than 1 and closer to 2
    data = [ [ 1.7, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 5: greater than 1 and closer to 1
    data = [ [ 1.2, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 6: greater than 1 and in the middle
    data = [ [ 1.5, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_cycleCountingAccordingToBinSize_twoPairsDefaultBinSize_countDepends():
    # case 1: aggregate to two bins
    data = [ [ 0.2, 2.0 ], [ 1.2, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 2.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: aggregate to one bin
    data = [ [ 0.7, 2.0 ], [ 1.2, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: aggregate to one bin - lager than binSize
    data = [ [ 1.7, 2.0 ], [ 2.2, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 2.0, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_cycleCountingAccordingToBinSize_twoPairsSmallerBinSize_countDepends():
    # case 1: aggregate to two bins
    data = [ [ 0.3, 2.0 ], [ 0.9, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=0.5 )
    expectedRst = [ [ 0.0, 0.0 ], [ 0.5, 2.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: aggregate to one bin
    data = [ [ 0.3, 2.0 ], [ 0.7, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=0.5 )
    expectedRst = [ [ 0.0, 0.0 ], [ 0.5, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_cycleCountingAccordingToBinSize_twoPairsLargerBinSize_countDepends():
    # case 1: aggregate to two bins
    data = [ [ 0.3, 2.0 ], [ 2.9, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=2.0 )
    expectedRst = [ [ 0.0, 2.0 ], [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: aggregate to one bin
    data = [ [ 1.8, 2.0 ], [ 2.7, 2.0 ] ]
    calRst = utils.cycleCountingAccordingToBinSize( data, binSize=2.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 2.0, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )