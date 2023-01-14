#!/usr/bin/env python3

from ffpack import lcc
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test johannessonMinMaxCounting function
###############################################################################
def test_johannessonMinMaxCounting_noPointsOrOnePoint_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, True )

    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, True )


def test_johannessonMinMaxCounting_incorrectDataDim_valueError():
    data = [ [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, True )

    data = [ [ [ 1.0 ] ] ]
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, False )
    
    with pytest.raises( ValueError ):
        _ = lcc.johannessonMinMaxCounting( data, True )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_twoPoints_empty( mock_get ):
    data = [ 0.0, 2.0 ]
    mock_get.return_value = [ 0.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_threePointsIncreasing_empty( mock_get ):
    data = [ 0.0, 2.0, 3.0 ]
    mock_get.return_value = [ 0.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_threePointsDecreasing_empty( mock_get ):
    data = [ 3.0, 2.0, 0.0 ]
    mock_get.return_value = [ 3.0, 0.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_threePointsDownward_empty( mock_get ):
    # case 1: left is lower
    data = [ 1.0, 0.0, 2.0 ]
    mock_get.return_value = [ 1.0, 0.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: right is lower
    data = [ 2.0, 0.0, 1.0 ]
    mock_get.return_value = [ 2.0, 0.0, 1.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_threePointsUpward_leftDistance( mock_get ):
    data = [ 0.0, 2.0, 1.0 ]
    mock_get.return_value = [ 0.0, 2.0, 1.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ 0.0, 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_fourPointsNoCrossing_leftDistance( mock_get ):
    # case 1: higher valley in the right
    data = [ 0.0, 3.0, 1.0, 2.0 ]
    mock_get.return_value = [ 0.0, 3.0, 1.0, 2.0 ]
    
    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ 0.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: higher valley in the left
    data = [ 1.0, 3.0, 0.0, 2.0 ]
    mock_get.return_value = [ 1.0, 3.0, 0.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ 1.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_fivePoints_aggrated( mock_get ):
    data = [ 1.0, 4.0, 0.0, 3.0, 2.0 ]
    mock_get.return_value = [ 1.0, 4.0, 0.0, 3.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ 1.0, 4.0, 1 ], [ 0.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 3.0, 2 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_withRedundencePoints_aggrated( mock_get ):
    data = [ 1.0, 2.0, 4.0, 3.0, 2.0, 0.0, 2.0, 3.0, 2.0 ]
    mock_get.return_value = [ 1.0, 4.0, 0.0, 3.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ 1.0, 4.0, 1 ], [ 0.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 3.0, 2 ] ]
    np.testing.assert_allclose( calRst, expectedRst )



@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_withSamePeakPoints_oneKept( mock_get ):
    data = [ 1.0, 2.0, 4.0, 4.0, 3.0, 2.0, 0.0, 2.0, 3.0, 3.0, 2.0 ]
    mock_get.return_value = [ 1.0, 4.0, 0.0, 3.0, 2.0 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ 1.0, 4.0, 1 ], [ 0.0, 3.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 3.0, 2 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakValleyFilter" )
def test_johannessonMinMaxCounting_normalUseCase_pass( mock_get ):
    # Standard level corssing counting data from E1049-85(2017) Fig.2(a)
    data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
             -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    mock_get.return_value = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
                              -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]

    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ -0.8, 1.3, 1 ], [ -0.8, 3.4, 1 ], [ 0.7, 2.5, 1 ], 
                    [ -1.4, -0.5, 1 ], [ -2.3, -2.2, 1 ], [ -2.6, -2.4, 1 ], 
                    [ -3.3, 1.5, 1 ], [ -3.3, 3.4, 1 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 0.1, 1 ], [ 0.2, 1 ], [ 0.9, 1 ], [ 1.8, 1 ], 
                    [ 2.1, 1 ], [ 4.2, 1 ], [ 4.8, 1 ], [ 6.7, 1] ] 
    np.testing.assert_allclose( calRst, expectedRst )


    # Slightly bias the data without change the slope
    data = [ -0.8, 0.2, 1.1, 1.3, 0.7, 0.8, 2.3, 3.4, 3.2, 1.8, 
             1.0, 0.7, 0.8, 2.1, 2.5, 1.8, 0.9, 0.0, -1.2, -1.4, 
             -0.5, -0.9, -1.8, -2.3, -2.2, -2.6, -2.4, -3.1, -3.3, 
             -2.2, -1.3, 0.1, 0.8, 1.5, 0.7, 1.3, 2.1, 2.8, 3.4, 
             2.7, 2.1, 0.8, 0.1, -0.5 ]
    mock_get.return_value = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
                              -2.2, -2.6, -2.4, -3.3, 1.5, 0.7, 3.4, -0.5 ]
    
    calRst = lcc.johannessonMinMaxCounting( data, False )
    expectedRst = [ [ -0.8, 1.3, 1 ], [ -0.8, 3.4, 1 ], [ 0.7, 2.5, 1 ], 
                    [ -1.4, -0.5, 1 ], [ -2.3, -2.2, 1 ], [ -2.6, -2.4, 1 ], 
                    [ -3.3, 1.5, 1 ], [ -3.3, 3.4, 1 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.johannessonMinMaxCounting( data, True )
    expectedRst = [ [ 0.1, 1 ], [ 0.2, 1 ], [ 0.9, 1 ], [ 1.8, 1 ], 
                    [ 2.1, 1 ], [ 4.2, 1 ], [ 4.8, 1 ], [ 6.7, 1] ] 
    np.testing.assert_allclose( calRst, expectedRst )
