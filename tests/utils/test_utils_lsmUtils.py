#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest

###############################################################################
# Test countingRstToCountingMatrix
###############################################################################
def test_countingRstToCountingMatrix_incorrectDim_valueError():
    # Test edge cases for empty list
    countingRst = [ ]
    with pytest.raises( ValueError ):
        _ = utils.countingRstToCountingMatrix( countingRst )

    countingRst = [ [ [ ] ] ]
    with pytest.raises( ValueError ):
        _ = utils.countingRstToCountingMatrix( countingRst )

    countingRst = [ [ 1.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = utils.countingRstToCountingMatrix( countingRst )


def test_countingRstToCountingMatrix_emptyInput_empty():
    countingRst = [ [ ] ]
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ ] ]
    expectedKeys = [ ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_onePoint_2dMatrix():
    countingRst = [ [ 2.0, 2.5, 1 ] ]
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 1.0 ],
                       [ 0.0, 0.0 ] ]
    expectedKeys = [ 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )


def test_countingRstToCountingMatrix_twoPoint_matrixDepends():
    # case 1: duplicate points
    countingRst = [ [ 2.0, 2.5, 1 ], [ 2.0, 2.5, 0.5 ] ]
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
    calKeys = [ float( i ) for i in calKeys ]
    expectedMatrix = [ [ 0.0, 1.5 ],
                       [ 0.0, 0.0 ] ]
    expectedKeys = [ 2.0, 2.5 ]
    np.testing.assert_allclose( calMatrix, expectedMatrix )
    np.testing.assert_allclose( calKeys, expectedKeys )

    # case 2: one same point
    countingRst = [ [ 1.0, 2.5, 1 ], [ 2.0, 2.5, 0.5 ] ]
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
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
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
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
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
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
    calMatrix, calKeys = utils.countingRstToCountingMatrix( countingRst )
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
