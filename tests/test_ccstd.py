#!/usr/bin/env python3

from ffpack import ccstd
import numpy as np
import pytest

###############################################################################
# Test levelCorssingCounting function
###############################################################################
def test_levelCrossingCounting_case1():
    # Standard level corssing counting data from E1049-85(2017) Fig.2(a)
    # No levels for this test case
    data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
             -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ -3.0, 1.0 ], [ -2.0, 1.0 ], [ -1.0, 2.0 ], [ 0.0, 2.0 ], 
                    [ 1.0, 5.0 ], [ 2.0, 3.0 ], [ 3.0, 2.0 ] ] 
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = [ -0.8, 0.0, 1.0, 1.3, 0.7, 1.0, 2.0, 3.4, 3.0, 2.0, 
             1.0, 0.7, 1.0, 2.0, 2.5, 2.0, 1.0, 0.0, -1.0, -1.4, 
             -0.5, -1.0, -2.0, -2.3, -2.2, -2.6, -2.4, -3.0, -3.3, 
             -2.0, -1.0, 0.0, 1.0, 1.5, 0.6, 1.0, 2.0, 3.0, 3.4, 
             3.0, 2.0, 1.0, 0.0, -0.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ -3.0, 1.0 ], [ -2.0, 1.0 ], [ -1.0, 2.0 ], [ 0.0, 2.0 ], 
                    [ 1.0, 5.0 ], [ 2.0, 3.0 ], [ 3.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Slightly bias the data without change the slope
    data = [ -0.8, 0.2, 1.1, 1.3, 0.7, 0.8, 2.3, 3.4, 3.2, 1.8, 
             1.0, 0.7, 0.8, 2.1, 2.5, 1.8, 0.9, 0.0, -1.2, -1.4, 
             -0.5, -0.9, -1.8, -2.3, -2.2, -2.6, -2.4, -3.1, -3.3, 
             -2.2, -1.3, 0.1, 0.8, 1.5, 0.7, 1.3, 2.1, 2.8, 3.4, 
             2.7, 2.1, 0.8, 0.1, -0.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ -3.0, 1.0 ], [ -2.0, 1.0 ], [ -1.0, 2.0 ], [ 0.0, 2.0 ], 
                    [ 1.0, 5.0 ], [ 2.0, 3.0 ], [ 3.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = [ 0.0, 2.0 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 2.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = [ -0.5, 2.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 2.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 3
    data = [ 2.0, 0.0 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 4
    data = [ 2.5, -0.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )
    
    # Trivial case 5
    data = [ 2.5, -1.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ -1.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 6
    data = [ 0.0, -2.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ -2.0, 1.0 ] , [ -1.0, 1.0 ]]
    np.testing.assert_allclose( calRst, expectedRst )
    
    # Trivial case 7
    data = [ -2.5, -0.5 ]
    calRst = ccstd.levelCrossingCounting( data )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_levelCrossingCounting_case2():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = ccstd.levelCrossingCounting( data )

    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = ccstd.levelCrossingCounting( data )

    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = ccstd.levelCrossingCounting( data )

def test_levelCrossingCounting_case3():
    # With levels

    # Standard level corssing counting data from E1049-85(2017) Fig.2(a)
    # Set the levels
    data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
             -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    levels = np.array( [ -3.0, -2.0, 2.0, 3.0 ] )
    calRst = ccstd.levelCrossingCounting( data, levels=levels )
    expectedRst = [ [ -3.0, 1.0 ], [ -2.0, 1.0 ], [ 2.0, 3.0 ], [ 3.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Add other levels
    data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
             -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    levels = [ -5, -4, -3, -2, 2, 3, 4, 5 ]
    calRst = ccstd.levelCrossingCounting( data, levels=levels )
    expectedRst = [ [ -3.0, 1.0 ], [ -2.0, 1.0 ], [ 2.0, 3.0 ], [ 3.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = [ 0.0, 2.2 ]
    levels = [ 0.0, 1.0 ]
    calRst = ccstd.levelCrossingCounting( data, levels=levels )
    expectedRst = [ [ 0.0, 1.0 ], [ 1.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = [ 0.0, 2.2 ]
    levels = [ 2.0 ]
    calRst = ccstd.levelCrossingCounting( data, levels=levels )
    expectedRst = [ [ 2.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 3
    data = [ 1.0, -2.5 ]
    levels = [ -1.0 ]
    calRst = ccstd.levelCrossingCounting( data, levels=levels )
    expectedRst = [ [ -1.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

###############################################################################
# Test peakCounting function
###############################################################################
def test_peakPeakCounting_case1():
    # Standard peak counting data from E1049-85(2017) Fig.3(a)
    data = [ 0.0, 1.5, 0.5, 3.5, 0.5, 2.5, -1.5, -0.5, -2.5, 
             -2.0, -2.7, -2.5, -3.5, 1.5, 0.5, 3.5, -0.5 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ -3.5, 1.0 ], [ -2.7, 1.0 ], [ -2.5, 1.0 ], [ -1.5, 1.0 ], 
                    [ 1.5, 2.0 ], [ 2.5, 1.0 ], [ 3.5, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = [ 0.0, 1.0, 1.5, 0.5, 0.5, 1.0, 2.0, 3.5, 2.0, 1.0, 0.5, 1.5, 2.5, 
             0.0, -1.0, -1.5, -1.0, -0.5, -2.0, -2.5, -2.0, -2.7, -2.5, -3.5, 
             -2.5, -0.5, 1.0, 1.5, 1.0, 0.5, 2.0, 3.5, 1.5, -0.5 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ -3.5, 1.0 ], [ -2.7, 1.0 ], [ -2.5, 1.0 ], [ -1.5, 1.0 ], 
                    [ 1.5, 2.0 ], [ 2.5, 1.0 ], [ 3.5, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = [ 0.0, 1.0 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = [ 0.0, 1.0, 0.0 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ 1.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 3
    data = [ 0.5, 1.0, -1.0 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ 1.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 4
    data = [ 0.5, -1.5, 1.0 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ -1.5, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 4
    data = [ 0.5, -1.5, 1.0, -0.5 ]
    calRst = ccstd.peakCounting( data )
    expectedRst = [ [ -1.5, 1.0 ], [ 1.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_peakCounting_case2():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = ccstd.peakCounting( data )

    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = ccstd.peakCounting( data )

    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = ccstd.peakCounting( data )

def test_peakPeakCounting_case3():
    # With level

    # Standard peak counting data from E1049-85(2017) Fig.3(a)
    data = [ 0.0, 1.5, 0.5, 3.5, 0.5, 2.5, -1.5, -0.5, -2.5, 
             -2.0, -2.7, -2.5, -3.5, 1.5, 0.5, 3.5, -0.5 ]
    # Set level to 1.0
    level = 1.0
    calRst = ccstd.peakCounting( data, level=level )
    expectedRst = [ [ -3.5, 1.0 ], [ -2.7, 1.0 ], [ -2.5, 1.0 ], [ -1.5, 1.0 ], 
                    [ 0.5, 3.0], [ 1.5, 2.0 ], [ 2.5, 1.0 ], [ 3.5, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Standard peak counting data from E1049-85(2017) Fig.3(a)
    data = [ 0.0, 1.5, 0.5, 3.5, 0.5, 2.5, -1.5, -0.5, -2.5, 
             -2.0, -2.7, -2.5, -3.5, 1.5, 0.5, 3.5, -0.5 ]
    # Set level to 2.0
    level = 2.0
    calRst = ccstd.peakCounting( data, level=level )
    expectedRst = [ [ -3.5, 1.0 ], [ -2.7, 1.0 ], [ -2.5, 1.0 ], [ -1.5, 1.0 ], 
                    [ 0.5, 3.0], [ 2.5, 1.0 ], [ 3.5, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Standard peak counting data from E1049-85(2017) Fig.3(a)
    data = [ 0.0, 1.5, 0.5, 3.5, 0.5, 2.5, -1.5, -0.5, -2.5, 
             -2.0, -2.7, -2.5, -3.5, 1.5, 0.5, 3.5, -0.5 ]
    # Set level to -1.0
    level = -1.0
    calRst = ccstd.peakCounting( data, level=level )
    expectedRst = [ [ -3.5, 1.0 ], [ -2.7, 1.0 ], [ -2.5, 1.0 ], [ -1.5, 1.0 ], 
                    [ -0.5, 1.0 ], [ 1.5, 2.0 ], [ 2.5, 1.0 ], [ 3.5, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Standard peak counting data from E1049-85(2017) Fig.3(a)
    data = [ 0.0, 1.5, 0.5, 3.5, 0.5, 2.5, -1.5, -0.5, -2.5, 
             -2.0, -2.7, -2.5, -3.5, 1.5, 0.5, 3.5, -0.5 ]
    # Set level to -1.8
    level = -1.8
    calRst = ccstd.peakCounting( data, level=level )
    expectedRst = [ [ -3.5, 1.0 ], [ -2.7, 1.0 ], [ -2.5, 1.0 ], [ -0.5, 1.0 ], 
                    [ 1.5, 2.0 ], [ 2.5, 1.0 ], [ 3.5, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = [ 0.0, 2.0, -0.5 ]
    # Set level to 1.0
    level = 1.0
    calRst = ccstd.peakCounting( data, level=level )
    expectedRst = [ [ 2.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = [ 0.5, -1.5, -0.5 ]
    # Set level to -1.0
    level = -1.0
    calRst = ccstd.peakCounting( data, level=level )
    expectedRst = [ [ -1.5, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

###############################################################################
# Test simpleRangeCounting function
###############################################################################
def test_simpleRangeCounting_case1():
    # Standard simple range counting data from E1049-85(2017) Fig.4(a)
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 3.0, 0.5 ], [ 4.0, 1.0 ], [ 6.0, 1.0 ], 
                    [ 7.0, 0.5 ], [ 8.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = [ -2.0, -0.5, 1.0, 0.0, -1.5, -3.0, -1.0, 2.5, 5.0, 3.0, 1.0, -0.5, -1.0, 2.0, 
             3.0, -0.5, -3.5, -4.0, 2.0, 3.0, 4.0, 3.0, 1.0, -2.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 3.0, 0.5 ], [ 4.0, 1.0 ], [ 6.0, 1.0 ], 
                    [ 7.0, 0.5 ], [ 8.0, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = [ 0.0, 1.5 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 1.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = [ 1.5, 2.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 0.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 3
    data = [ 3.0, 1.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 2.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 4
    data = [ -0.5, -1.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 0.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 5
    data = [ 0.0, 1.5, 0.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 1.5, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 6
    data = [ -1.0, -2.5, -1.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 1.5, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 7
    data = [ 0.0, 1.5, 1.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 0.5, 0.5 ], [ 1.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 8
    data = [ 0.0, -2.5, -1.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 1.5, 0.5 ], [ 2.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 9
    data = [ 0.0, 2.5, 0.0, 3.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 2.5, 1.0 ], [ 3.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 10
    data = [ -1.0, 3.0, -0.5, 1.0, -2.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 1.5, 0.5 ], [ 3.0, 0.5 ], [ 3.5, 0.5 ], [ 4.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 11
    data = [ -1.0, 1.0, -3.0, 3.0, 2.0 ]
    calRst = ccstd.simpleRangeCounting( data )
    expectedRst = [ [ 1.0, 0.5 ], [ 2.0, 0.5 ], [ 4.0, 0.5 ], [ 6.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_simpleRangeCounting_case2():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = ccstd.simpleRangeCounting( data )

    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = ccstd.simpleRangeCounting( data )

    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = ccstd.simpleRangeCounting( data )

###############################################################################
# Test rainflowCounting function
###############################################################################
def test_rainflowCounting_case1():
    # Standard rainflow counting data from E1049-85(2017) Fig.6(a)
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 3.0, 0.5 ], [ 4.0, 1.5 ], [ 6.0, 0.5 ], [ 8.0, 1.0 ], [ 9.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = [ -2.0, -0.5, 1.0, 0.0, -1.5, -3.0, -1.0, 2.5, 5.0, 3.0, 1.0, -0.5, -1.0, 2.0, 
             3.0, -0.5, -3.5, -4.0, 2.0, 3.0, 4.0, 3.0, 1.0, -2.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 3.0, 0.5 ], [ 4.0, 1.5 ], [ 6.0, 0.5 ], [ 8.0, 1.0 ], [ 9.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = [ 0.0, 1.5 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 1.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = [ 1.5, 2.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 0.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 3
    data = [ 3.0, 1.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 2.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 4
    data = [ -0.5, -1.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 0.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 5
    data = [ 0.0, 1.5, 0.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 1.5, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 6
    data = [ -1.0, -2.5, -1.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 1.5, 1.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 7
    data = [ 0.0, 1.5, 1.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 0.5, 0.5 ], [ 1.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 8
    data = [ 0.0, -2.5, -1.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 1.5, 0.5 ], [ 2.5, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 9
    data = [ 0.0, 2.5, 0.0, 3.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 2.5, 1.0 ], [ 3, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 10
    data = [ -1.0, 3.0, -0.5, 1.0, -2.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 1.5, 1.0 ], [ 4.0, 0.5 ], [ 5.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 11
    data = [ -1.0, 1.0, -3.0, 3.0, 2.0 ]
    calRst = ccstd.rainflowCounting( data )
    expectedRst = [ [ 1.0, 0.5 ], [ 2.0, 0.5 ], [ 4.0, 0.5 ], [ 6.0, 0.5 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

def test_rainflowCounting_case2():
    # Test edge cases for empty list
    data = [ ]
    with pytest.raises( ValueError ):
        _ = ccstd.rainflowCounting( data )

    # Test edge cases for 1 element list
    data = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = ccstd.rainflowCounting( data )

    # Test edge cases for 2D list
    data = [ [ 1.0 ], [ 2.0 ] ]
    with pytest.raises( ValueError ):
        _ = ccstd.rainflowCounting( data )
