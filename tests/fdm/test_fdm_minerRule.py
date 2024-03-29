#!/usr/bin/env python3

from ffpack import fdm
import numpy as np
import pytest
from unittest.mock import patch
from ffpack.utils import SnCurveFitter


###############################################################################
# Test minerDamageModelNaive
###############################################################################
def test_minerDamageModelNaive_emptyInput_valueError():
    fatigueData = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelNaive( fatigueData )


def test_minerDamageModelNaive_oneDimInput_valueError():
    fatigueData = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelNaive( fatigueData )


def test_minerDamageModelNaive_irregularInput_valueError():
    fatigueData = [ [ -10, 100 ], [ 200, -2000 ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelNaive( fatigueData )

    fatigueData = [ [ 10, 100 ], [ 0, 0 ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelNaive( fatigueData )

    fatigueData = [ [ 10, 100 ], [ 2001, 2000 ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelNaive( fatigueData )


def test_minerDamageModelNaive_twoPairs_scalarOutput():
    fatigueData = [ [ 10, 100 ], [ 200, 2000 ] ]
    calRst = fdm.minerDamageModelNaive( fatigueData )
    expectedRst = 0.2 
    np.testing.assert_allclose( calRst, expectedRst )


def test_minerDamageModelNaive_threePairs_scalarOutput():
    fatigueData = [ [ 10, 1000 ], [ 200, 20000 ], [ 50, 500 ] ]
    calRst = fdm.minerDamageModelNaive( fatigueData )
    expectedRst = 0.12
    np.testing.assert_allclose( calRst, expectedRst )


def test_minerDamageModelNaive_fourPairs_scalarOutput():
    fatigueData = [ [ 1, 100 ], [ 2, 2000 ], [ 3, 30 ], [ 4, 40 ] ]
    calRst = fdm.minerDamageModelNaive( fatigueData )
    expectedRst = 0.211 
    np.testing.assert_allclose( calRst, expectedRst )


def test_minerDamageModelNaive_fourPairsLargeCounts_scalarOutput():
    fatigueData = [ [ 50, 100 ], [ 1000, 2000 ], [ 15, 30 ], [ 40, 40 ] ]
    calRst = fdm.minerDamageModelNaive( fatigueData )
    expectedRst = 2.5 
    np.testing.assert_allclose( calRst, expectedRst )


###############################################################################
# Test minerDamageModelClassic
###############################################################################
def test_minerDamageModelClassic_emptyInput_valueError():
    lccData = [ [ ] ] 
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )

    lccData = [ ] 
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )


def test_minerDamageModelClassic_oneDimInput_valueError():
    lccData = [ 1.0, 2.0, 3.0 ] 
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )


def test_minerDamageModelClassic_irregularInput_valueError():
    lccData = [ [ 1, 1000 ], [ -2, 100 ], [ 4, 10 ] ]
    snData = [ [ 10, 5 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )

    lccData = [ [ 1, 1000 ], [ 2, -100 ], [ 4, 10 ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )

    lccData = [ [ 1, 1000 ], [ 0, 100 ], [ 4, 10 ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )

    lccData = [ [ 1, 1000 ], [ 2, 0 ], [ 4, 10 ] ]
    with pytest.raises( ValueError ):
        _ = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )


@patch.object( SnCurveFitter, "getN" )
def test_minerDamageModelClassic_twoPairs_scalarOutput( mocker ):

    mocker.side_effect = lambda x: { 1: 1000, 2: 100 }[ x ]

    lccData = [ [ 1, 100 ], [ 2, 10 ] ]
    snData = [ [ 10, 3 ], [ 1000, 1 ] ]
    fatigueLimit = 0.5
    
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.2 
    np.testing.assert_allclose( calRst, expectedRst )


@patch.object( SnCurveFitter, "getN" )
def test_minerDamageModelClassic_threePairs_scalarOutput( mocker ):

    mocker.side_effect = lambda x: { 1: 100000, 2: 10000, 3: 1000, 4: 100 }[ x ]

    lccData = [ [ 1, 1000 ], [ 2, 100 ], [ 4, 10 ] ]
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.12 
    np.testing.assert_allclose( calRst, expectedRst )

    lccData = [ [ 1, 1000 ], [ 3, 100 ], [ 4, 10 ] ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.21 
    np.testing.assert_allclose( calRst, expectedRst )

    lccData = [ [ 1, 1000 ], [ 3, 100 ], [ 4, 100 ] ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 1.11 
    np.testing.assert_allclose( calRst, expectedRst )

    lccData = [ [ 1, 1000 ], [ 3, 1000 ], [ 4, 100 ] ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 2.01 
    np.testing.assert_allclose( calRst, expectedRst )


@patch.object( SnCurveFitter, "getN" )
def test_minerDamageModelClassic_threePairsHighFatigueLimit_scalarOutput( mocker ):
    lccData = [ [ 1, 1000 ], [ 2, 100 ], [ 4, 10 ] ]
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]

    fatigueLimit = 1  
    mocker.side_effect = lambda x: { 1: -1, 2: 10000, 4: 100 }[ x ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.11 
    np.testing.assert_allclose( calRst, expectedRst )

    fatigueLimit = 2
    mocker.side_effect = lambda x: { 1: -1, 2: -1, 4: 100 }[ x ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.1 
    np.testing.assert_allclose( calRst, expectedRst )

    fatigueLimit = 3
    mocker.side_effect = lambda x: { 1: -1, 2: -1, 4: 100 }[ x ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.1 
    np.testing.assert_allclose( calRst, expectedRst )

    fatigueLimit = 4
    mocker.side_effect = lambda x: { 1: -1, 2: -1, 4: -1 }[ x ]
    calRst = fdm.minerDamageModelClassic( lccData, snData, fatigueLimit )
    expectedRst = 0
    np.testing.assert_allclose( calRst, expectedRst )
