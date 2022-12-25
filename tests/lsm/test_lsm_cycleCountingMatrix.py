#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest
from unittest.mock import Mock
from ffpack import utils


###############################################################################
# Test astmSimpleRangeCountingMatrix
###############################################################################
def test_astmSimpleRangeCountingMatrix_severalPoints_2dMatrix():
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmSimpleRangeCountingMatrix( data, digitization=True, resolution=1.0 )
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
# Test astmRainflowCountingMatrix
###############################################################################
def test_astmRainflowCountingMatrix_severalPoints_2dMatrix():
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.astmRainflowCountingMatrix( data, digitization=True, resolution=1.0 )
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



###############################################################################
# Test rychlikRainflowCountingmatrix
###############################################################################
def test_rychlikRainflowCountingmatrix_severalPoints_2dMatrix():
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calMatrix, calKeys = lsm.rychlikRainflowCountingmatrix( data, digitization=True, resolution=1.0 )
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
