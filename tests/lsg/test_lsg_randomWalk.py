#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest

###############################################################################
# Test randomWalkUniform
###############################################################################
def test_randomWalkUniform_normalUseCase_diffByOne():
    calRst = lsg.randomWalkUniform( 100, dim=1 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalkUniform( 100, dim=2 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalkUniform( 100, dim=3 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalkUniform( 100, dim=4 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

    calRst = lsg.randomWalkUniform( 50, dim=8 )
    sumRowRst = np.sum( np.array( calRst ), axis=1 ).astype( int )
    for i in range( 1, len( sumRowRst ) ):
        assert abs( sumRowRst[ i ] - sumRowRst[ i - 1 ] ) == 1

def test_randomWalkUniform_stepsLessThanOneCase_valueError():
    with pytest.raises( ValueError ):
        _ = lsg.randomWalkUniform( 0 )

    with pytest.raises( ValueError ):
        _ = lsg.randomWalkUniform( -2 )

def test_randomWalkUniform_dimLessThanOneCase_valueError():
    steps = 1
    dim = 0
    with pytest.raises( ValueError ):
        _ = lsg.randomWalkUniform( 1, dim=0 )

    steps = 1
    dim = -2
    with pytest.raises( ValueError ):
        _ = lsg.randomWalkUniform( 1, dim=-2 )