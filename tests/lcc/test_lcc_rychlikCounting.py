#!/usr/bin/env python3

from ffpack import lcc
import numpy as np
import pytest

###############################################################################
# Test rychlikRainflowCycleCounting function
###############################################################################
def test_rychlikRainflowCycleCounting_twoPoints_empty():
    data = [ 0.0, 2.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_threePointsIncreasing_empty():
    data = [ 0.0, 2.0, 3.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_threePointsDecreasing_empty():
    data = [ 3.0, 2.0, 0.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_threePointsDownward_empty():
    # case 1: left is lower
    data = [ 1.0, 0.0, 2.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: right is lower
    data = [ 2.0, 0.0, 1.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_threePointsUpward_smallerDistance():
    # case 1: right is higher
    data = [ 0.0, 2.0, 1.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 1.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: left is higher
    data = [ 1.0, 2.0, 0.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 1.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 1.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_fourPointsNoCrossing_smallerDistance():
    # case 1: higher valley in the right
    data = [ 0.0, 3.0, 1.0, 2.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: higher valley in the left
    data = [ 1.0, 3.0, 0.0, 2.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_fivePoints_aggrated():
    data = [ 0.0, 3.0, 1.0, 4.0, 2.0 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 2.0, 2.0 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 2.0, 2 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_rychlikRainflowCycleCounting_normalUseCase_pass():
    # Standard level corssing counting data from E1049-85(2017) Fig.2(a)
    # No levels for this test case
    data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
             -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 1.3 - 0.7, 3.4 + 0.8, 2.5 - 0.7, -0.5 + 1.4, 
                    -2.2 + 2.3, -2.4 + 2.6, 1.5 - 0.6, 3.4 + 0.5] 
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 0.1, 1.0 ], [ 0.2, 1.0 ], [ 0.6, 1.0 ], 
             [ 0.9, 2.0 ], [ 1.8, 1.0 ], [ 3.9, 1.0 ], [ 4.2, 1.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = [ -0.8, 0.0, 1.0, 1.3, 0.7, 1.0, 2.0, 3.4, 3.0, 2.0, 
             1.0, 0.7, 1.0, 2.0, 2.5, 2.0, 1.0, 0.0, -1.0, -1.4, 
             -0.5, -1.0, -2.0, -2.3, -2.2, -2.6, -2.4, -3.0, -3.3, 
             -2.0, -1.0, 0.0, 1.0, 1.5, 0.6, 1.0, 2.0, 3.0, 3.4, 
             3.0, 2.0, 1.0, 0.0, -0.5 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 1.3 - 0.7, 3.4 + 0.8, 2.5 - 0.7, -0.5 + 1.4, 
                    -2.2 + 2.3, -2.4 + 2.6, 1.5 - 0.6, 3.4 + 0.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 0.1, 1.0 ], [ 0.2, 1.0 ], [ 0.6, 1.0 ], 
             [ 0.9, 2.0 ], [ 1.8, 1.0 ], [ 3.9, 1.0 ], [ 4.2, 1.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # Slightly bias the data without change the slope
    data = [ -0.8, 0.2, 1.1, 1.3, 0.7, 0.8, 2.3, 3.4, 3.2, 1.8, 
             1.0, 0.7, 0.8, 2.1, 2.5, 1.8, 0.9, 0.0, -1.2, -1.4, 
             -0.5, -0.9, -1.8, -2.3, -2.2, -2.6, -2.4, -3.1, -3.3, 
             -2.2, -1.3, 0.1, 0.8, 1.5, 0.7, 1.3, 2.1, 2.8, 3.4, 
             2.7, 2.1, 0.8, 0.1, -0.5 ]
    calRst = lcc.rychlikRainflowCycleCounting( data, False )
    expectedRst = [ 1.3 - 0.7, 3.4 + 0.8, 2.5 - 0.7, -0.5 + 1.4, 
                    -2.2 + 2.3, -2.4 + 2.6, 1.5 - 0.7, 3.4 + 0.5 ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCycleCounting( data, True )
    expectedRst = [ [ 0.1, 1.0 ], [ 0.2, 1.0 ], [ 0.6, 1.0 ], [ 0.8, 1.0 ],
             [ 0.9, 1.0 ], [ 1.8, 1.0 ], [ 3.9, 1.0 ], [ 4.2, 1.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )
