#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest

###############################################################################
# Test randomWalk
###############################################################################
def test_randomWalk_normalUseCase_diffByOne():
    calRst = lsg.randomWalk( 100, 1 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalk( 100, 2 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalk( 100, 3 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalk( 100, 4 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalk( 50, 8 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

def test_randomWalk_stepsLessThanOneCase_valueError():
    steps = 0
    with pytest.raises( ValueError ):
        _ = lsg.randomWalk( steps=steps )

    steps = -2
    with pytest.raises( ValueError ):
        _ = lsg.randomWalk( steps=steps )

def test_randomWalk_dimLessThanOneCase_valueError():
    steps = 1
    dim = 0
    with pytest.raises( ValueError ):
        _ = lsg.randomWalk( steps=steps, dim=dim )

    steps = 1
    dim = -2
    with pytest.raises( ValueError ):
        _ = lsg.randomWalk( steps=steps, dim=dim )