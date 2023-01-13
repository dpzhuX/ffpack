#!/usr/bin/env python3

from ffpack import lcc
import numpy as np
import pytest
from unittest.mock import patch


##############################################################################
# Test fourPointRainflowCounting function
###############################################################################
def test_fourPointRainflowCounting_emptyInputCase_valueError():
    data = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.fourPointRainflowCounting( data )


def test_fourPointRainflowCounting_lessInputCase_valueError():
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.fourPointRainflowCounting( data )
    
    data = [ 1.0, 2.0, 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.fourPointRainflowCounting( data )


def test_fourPointRainflowCounting_twoDimInputCase_valueError():
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = lcc.fourPointRainflowCounting( data )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_fourPointRainflowCounting_fourPoints_depends( mock_get ):
    # case 1: no cycle
    data = [ -1.0, 2.0, -2.0, 2.0 ]
    mock_get.return_value = data
    calRst = lcc.fourPointRainflowCounting( data, aggregate=False )
    expectedRst = [ ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.fourPointRainflowCounting( data, aggregate=True )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: one cycle
    data = [ -2.0, 1.0, -1.0, 1.0 ]
    mock_get.return_value = data
    calRst = lcc.fourPointRainflowCounting( data, aggregate=False )
    expectedRst = [ [ 1.0, -1.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.fourPointRainflowCounting( data, aggregate=True )
    expectedRst = [ [ 2.0, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_fourPointRainflowCounting_book_aggregated( mock_get ):
    data = [ 2, -1, 3, -5, 1, -3, 4, -4, 2 ]
    mock_get.return_value = data
    calRst = lcc.fourPointRainflowCounting( data, aggregate=False )
    expectedRst = [ [ 1, -3, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.fourPointRainflowCounting( data, aggregate=True )
    expectedRst = [ [ 4, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


@patch( "ffpack.utils.generalUtils.sequencePeakAndValleys" )
def test_fourPointRainflowCounting_web_aggregated( mock_get ):
    data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 4, 1, 5, 3, 6, 3, 6, 1, 5, 2 ]
    mock_get.return_value = data
    calRst = lcc.fourPointRainflowCounting( data, aggregate=False )
    expectedRst = [ [ 5, 3, 1 ], [ 2, 4, 1 ], [ 1, 6, 1 ], [ 1, 4, 1 ], [ 5, 3, 1 ], 
                    [ 6, 3, 1 ], [ 1, 6, 1 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    calRst = lcc.fourPointRainflowCounting( data, aggregate=True )
    expectedRst = [ [ 2, 3 ], [ 3, 2 ], [ 5, 2 ] ]
    np.testing.assert_allclose( calRst, expectedRst )
