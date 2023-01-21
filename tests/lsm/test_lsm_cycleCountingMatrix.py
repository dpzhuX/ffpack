#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest


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



###############################################################################
# Test fourPointCountingMatrix
###############################################################################
def test_fourPointCountingMatrix_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lsm.fourPointCountingMatrix( data )


def test_fourPointCountingMatrix_singleInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lsm.fourPointCountingMatrix( data )


def test_fourPointCountingMatrix_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lsm.fourPointCountingMatrix( data )
