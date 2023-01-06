#!/usr/bin/env python3

from ffpack import lcc
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test rychlikRainflowCounting function
###############################################################################
def test_rychlikRainflowCounting_noPointsOrOnePoint_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, True )

    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, True )


def test_rychlikRainflowCounting_incorrectDataDim_valueError():
    data = [ [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, True )

    data = [ [ [ 1.0 ] ] ]
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.rychlikRainflowCounting( data, True )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_twoPoints_empty( mock_get ):
    data = [ 0.0, 2.0 ]
    mock_get.return_value = [ 0.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_threePointsIncreasing_empty( mock_get ):
    data = [ 0.0, 2.0, 3.0 ]
    mock_get.return_value = [ 0.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_threePointsDecreasing_empty( mock_get ):
    data = [ 3.0, 2.0, 0.0 ]
    mock_get.return_value = [ 3.0, 0.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_threePointsDownward_empty( mock_get ):
    # case 1: left is lower
    data = [ 1.0, 0.0, 2.0 ]
    mock_get.return_value = [ 1.0, 0.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: right is lower
    data = [ 2.0, 0.0, 1.0 ]
    mock_get.return_value = [ 2.0, 0.0, 1.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_threePointsUpward_smallerDistance( mock_get ):
    # case 1: right is higher
    data = [ 0.0, 2.0, 1.0 ]
    mock_get.return_value = [ 0.0, 2.0, 1.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 1.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: left is higher
    data = [ 1.0, 2.0, 0.0 ]
    mock_get.return_value = [ 1.0, 2.0, 0.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 1.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_fourPointsNoCrossing_smallerDistance( mock_get ):
    # case 1: higher valley in the right
    data = [ 0.0, 3.0, 1.0, 2.0 ]
    mock_get.return_value = [ 0.0, 3.0, 1.0, 2.0 ]
    
    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: higher valley in the left
    data = [ 1.0, 3.0, 0.0, 2.0 ]
    mock_get.return_value = [ 1.0, 3.0, 0.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_fivePoints_aggrated( mock_get ):
    data = [ 0.0, 3.0, 1.0, 4.0, 2.0 ]
    mock_get.return_value = [ 0.0, 3.0, 1.0, 4.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 3.0, 1 ], [ 2.0, 4.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_withRedundencePoints_aggrated( mock_get ):
    data = [ 0.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0 ]
    mock_get.return_value = [ 0.0, 3.0, 1.0, 4.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 3.0, 1 ], [ 2.0, 4.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_withSamePeakPoints_oneKept( mock_get ):
    data = [ 0.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 2.0 ]
    mock_get.return_value = [ 0.0, 3.0, 1.0, 4.0, 2.0 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 1.0, 3.0, 1 ], [ 2.0, 4.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_rychlikRainflowCounting_normalUseCase_pass( mock_get ):
    # Standard level corssing counting data from E1049-85(2017) Fig.2(a)
    # No levels for this test case
    data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
             -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    mock_get.return_value = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
                              -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]

    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 0.7, 1.3, 1 ], [ -0.8, 3.4, 1 ], [ 0.7, 2.5, 1 ], 
                    [ -1.4, -0.5, 1 ], [ -2.3, -2.2, 1 ], [ -2.6, -2.4, 1 ], 
                    [ 0.6, 1.5, 1 ], [ -0.5, 3.4, 1 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 0.1, 1.0 ], [ 0.2, 1.0 ], [ 0.6, 1.0 ], 
                    [ 0.9, 2.0 ], [ 1.8, 1.0 ], [ 3.9, 1.0 ], [ 4.2, 1.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = [ -0.8, 0.0, 1.0, 1.3, 0.7, 1.0, 2.0, 3.4, 3.0, 2.0, 
             1.0, 0.7, 1.0, 2.0, 2.5, 2.0, 1.0, 0.0, -1.0, -1.4, 
             -0.5, -1.0, -2.0, -2.3, -2.2, -2.6, -2.4, -3.0, -3.3, 
             -2.0, -1.0, 0.0, 1.0, 1.5, 0.6, 1.0, 2.0, 3.0, 3.4, 
             3.0, 2.0, 1.0, 0.0, -0.5 ]
    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 0.7, 1.3, 1 ], [ -0.8, 3.4, 1 ], [ 0.7, 2.5, 1 ], 
                    [ -1.4, -0.5, 1 ], [ -2.3, -2.2, 1 ], [ -2.6, -2.4, 1 ], 
                    [ 0.6, 1.5, 1 ], [ -0.5, 3.4, 1 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 0.1, 1.0 ], [ 0.2, 1.0 ], [ 0.6, 1.0 ], 
                    [ 0.9, 2.0 ], [ 1.8, 1.0 ], [ 3.9, 1.0 ], [ 4.2, 1.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # Slightly bias the data without change the slope
    data = [ -0.8, 0.2, 1.1, 1.3, 0.7, 0.8, 2.3, 3.4, 3.2, 1.8, 
             1.0, 0.7, 0.8, 2.1, 2.5, 1.8, 0.9, 0.0, -1.2, -1.4, 
             -0.5, -0.9, -1.8, -2.3, -2.2, -2.6, -2.4, -3.1, -3.3, 
             -2.2, -1.3, 0.1, 0.8, 1.5, 0.7, 1.3, 2.1, 2.8, 3.4, 
             2.7, 2.1, 0.8, 0.1, -0.5 ]
    mock_get.return_value = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
                              -2.2, -2.6, -2.4, -3.3, 1.5, 0.7, 3.4, -0.5 ]
    
    calRst = lcc.rychlikRainflowCounting( data, False )
    expectedRst = [ [ 0.7, 1.3, 1 ], [ -0.8, 3.4, 1 ], [ 0.7, 2.5, 1 ], 
                    [ -1.4, -0.5, 1 ], [ -2.3, -2.2, 1 ], [ -2.6, -2.4, 1 ], 
                    [ 0.7, 1.5, 1 ], [ -0.5, 3.4, 1 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.rychlikRainflowCounting( data, True )
    expectedRst = [ [ 0.1, 1.0 ], [ 0.2, 1.0 ], [ 0.6, 1.0 ], [ 0.8, 1.0 ],
                    [ 0.9, 1.0 ], [ 1.8, 1.0 ], [ 3.9, 1.0 ], [ 4.2, 1.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )
