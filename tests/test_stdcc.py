#!/usr/bin/env python3

from ffpack import stdcc
import numpy as np
import pytest

#
# Test levelCorssingCounting function
#
def test_levelCrossingCounting_case1():
    # Standard level corssing counting data from E1049-85(2017) Fig.2(a)
    data = np.array( [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
                       -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ -3, 1 ], [ -2, 1 ], [ -1, 2 ], [ 0, 2 ], 
                              [ 1, 5 ], [ 2, 3 ], [3, 2] ] )
    np.testing.assert_allclose( calRst, expectedRst )

    # Adding extra data into the standard data without change the slope
    data = np.array( [ -0.8, 0.0, 1.0, 1.3, 0.7, 1.0, 2.0, 3.4, 3.0, 2.0, 
                       1.0, 0.7, 1.0, 2.0, 2.5, 2.0, 1.0, 0.0, -1.0, -1.4, 
                       -0.5, -1.0, -2.0, -2.3, -2.2, -2.6, -2.4, -3.0, -3.3, 
                       -2.0, -1.0, 0.0, 1.0, 1.5, 0.6, 1.0, 2.0, 3.0, 3.4, 
                       3.0, 2.0, 1.0, 0.0, -0.5 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ -3, 1 ], [ -2, 1 ], [ -1, 2 ], [ 0, 2 ], 
                              [ 1, 5 ], [ 2, 3 ], [3, 2] ] )
    np.testing.assert_allclose( calRst, expectedRst )

    # Slightly bias the data without change the slope
    data = np.array( [ -0.8, 0.2, 1.1, 1.3, 0.7, 0.8, 2.3, 3.4, 3.2, 1.8, 
                       1.0, 0.7, 0.8, 2.1, 2.5, 1.8, 0.9, 0.0, -1.2, -1.4, 
                       -0.5, -0.9, -1.8, -2.3, -2.2, -2.6, -2.4, -3.1, -3.3, 
                       -2.2, -1.3, 0.1, 0.8, 1.5, 0.7, 1.3, 2.1, 2.8, 3.4, 
                       2.7, 2.1, 0.8, 0.1, -0.5 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ -3, 1 ], [ -2, 1 ], [ -1, 2 ], [ 0, 2 ], 
                              [ 1, 5 ], [ 2, 3 ], [3, 2] ] )
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 1
    data = np.array( [ 0.0, 2.0 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ 0, 1 ], [ 1, 1 ], [ 2, 1 ] ] )
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 2
    data = np.array( [ -0.5, 2.5 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ 0, 1 ], [ 1, 1 ], [ 2, 1 ] ] )
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 3
    data = np.array( [ 2.0, 0.0 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ ] ] )
    np.testing.assert_allclose( calRst, expectedRst )

    # Trivial case 4
    data = np.array( [ 2.5, -0.5 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ ] ] )
    np.testing.assert_allclose( calRst, expectedRst )
    
    # Trivial case 5
    data = np.array( [ 2.5, -1.5 ] )
    calRst = stdcc.levelCrossingCounting( data )
    expectedRst = np.array( [ [ -1, 1 ] ] )
    np.testing.assert_allclose( calRst, expectedRst )

def test_levelCrossingCounting_case2():
    # Test edge cases for empty list
    data = np.array( [ ] )
    with pytest.raises( ValueError ):
        _ = stdcc.levelCrossingCounting( data )

    # Test edge cases for 1 element list
    data = np.array( [ 1 ] )
    with pytest.raises( ValueError ):
        _ = stdcc.levelCrossingCounting( data )

    # Test edge cases for 2D list
    data = np.array( [ [ 1 ], [ 2 ] ] )
    with pytest.raises( ValueError ):
        _ = stdcc.levelCrossingCounting( data )