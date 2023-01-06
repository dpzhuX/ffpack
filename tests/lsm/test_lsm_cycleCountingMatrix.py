#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest


###############################################################################
# Test countingRstToCountingMatrix
###############################################################################
def test_countingRstToCountingMatrix_incorrectDim_valueError():
    # Test edge cases for empty list
    countingRst = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.countingRstToCountingMatrix( countingRst )

    countingRst = [ [ [ ] ] ]
    with pytest.raises( ValueError ):
        _ = lsm.countingRstToCountingMatrix( countingRst )

    countingRst = [ [ 1.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.countingRstToCountingMatrix( countingRst )


def test_countingRstToCountingMatrix_emptyInput_empty():
    countingRst = [ [ ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ ] ]
    expectedKeys = [ ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_onePoint_2dMatrix():
    countingRst = [ [ 2.0, 2.5, 1 ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 1.0 ],
                       [ 0.0, 0.0 ] ]
    expectedKeys = [ 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_twoPoint_matrixDepends():
    # case 1: duplicate points
    countingRst = [ [ 2.0, 2.5, 1 ], [ 2.0, 2.5, 0.5 ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 1.5 ],
                       [ 0.0, 0.0 ] ]
    expectedKeys = [ 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # case 2: one same point
    countingRst = [ [ 1.0, 2.5, 1 ], [ 2.0, 2.5, 0.5 ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 0.0, 1.0 ],
                       [ 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ 1.0, 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_fourPoint_matrixDepends():
    # case 1: with duplicate point
    countingRst = [ [ -2.0, 1.0, 1 ], [ -3.0, 5.0, 1 ], [ -1.0, 3.0, 1 ], 
                    [ -2.0, 4.0, 1 ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
                       [ 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_sevenPoint_matrixDepends():
    # case 1: with duplicate point
    countingRst = [ [ -2.0, 1.0, 0.5 ], [ 1.0, -3.0, 0.5 ], [ -1.0, 3.0, 1.0 ],
                    [ -3.0, 5.0, 0.5 ], [ 5.0, -4.0, 0.5 ], [ -4.0, 4.0, 0.5 ],
                    [ 4.0, -2.0, 0.5 ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_eightPoint_matrixDepends():
    # case 1: with duplicate point
    countingRst = [ [ -2.0, 1.0, 0.5 ], [ 1.0, -3.0, 0.5 ], [ -3.0, 5.0, 0.5 ], 
                    [ 5.0, -1.0, 0.5 ], [ -1.0, 3.0, 0.5 ], [ 3.0, -4.0, 0.5 ], 
                    [ -4.0, 4.0, 0.5 ], [ 4.0, -2.0, 0.5 ] ]
    calMatrix, calKeys = lsm.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0 ],
                       [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                       [ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 ] ]
    expectedKeys = [ -4.0, -3.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )



###############################################################################
# Test astmSimpleRangeCountingMatrix
###############################################################################
def test_astmSimpleRangeCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.astmSimpleRangeCountingMatrix( data )


def test_astmSimpleRangeCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.astmSimpleRangeCountingMatrix( data )


def test_astmSimpleRangeCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.astmSimpleRangeCountingMatrix( data )



###############################################################################
# Test astmRainflowCountingMatrix
###############################################################################
def test_astmRainflowCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowCountingMatrix( data )


def test_astmRainflowCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowCountingMatrix( data )


def test_astmRainflowCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowCountingMatrix( data )



###############################################################################
# Test astmRangePairCountingMatrix
###############################################################################
def test_astmRangePairCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRangePairCountingMatrix( data )


def test_astmRangePairCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRangePairCountingMatrix( data )


def test_astmRangePairCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRangePairCountingMatrix( data )



###############################################################################
# Test astmRainflowRepeatHistoryCountingMatrix
###############################################################################
def test_astmRainflowRepeatHistoryCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowRepeatHistoryCountingMatrix( data )


def test_astmRainflowRepeatHistoryCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowRepeatHistoryCountingMatrix( data )


def test_astmRainflowRepeatHistoryCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.astmRainflowRepeatHistoryCountingMatrix( data )



###############################################################################
# Test rychlikRainflowCountingmatrix
###############################################################################
def test_rychlikRainflowCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.rychlikRainflowCountingMatrix( data )


def test_rychlikRainflowCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.rychlikRainflowCountingMatrix( data )


def test_rychlikRainflowCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.rychlikRainflowCountingMatrix( data )



###############################################################################
# Test johannessonMinMaxCountingMatrix
###############################################################################
def test_johannessonMinMaxCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.johannessonMinMaxCountingMatrix( data )


def test_johannessonMinMaxCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.johannessonMinMaxCountingMatrix( data )


def test_johannessonMinMaxCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.johannessonMinMaxCountingMatrix( data )
