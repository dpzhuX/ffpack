#!/usr/bin/env python3

from ffpack import fdr
import numpy as np
import pytest

###############################################################################
# Test minerDamageRuleNaive
###############################################################################
def test_minerDamageRuleNaive_twoPairs_scalarOutput():
    fatigueData = [ [ 10, 100 ], [ 200, 2000 ] ]
    calRst = fdr.minerDamageRuleNaive( fatigueData )
    expectedRst = 0.2 
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleNaive_threePairs_scalarOutput():
    fatigueData = [ [ 10, 1000 ], [ 200, 20000 ], [ 50, 500 ] ]
    calRst = fdr.minerDamageRuleNaive( fatigueData )
    expectedRst = 0.12
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleNaive_fourPairs_scalarOutput():
    fatigueData = [ [ 1, 100 ], [ 2, 2000 ], [ 3, 30 ], [ 4, 40 ] ]
    calRst = fdr.minerDamageRuleNaive( fatigueData )
    expectedRst = 0.211 
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleNaive_fourPairs_scalarOutput():
    fatigueData = [ [ 50, 100 ], [ 1000, 2000 ], [ 15, 30 ], [ 40, 40 ] ]
    calRst = fdr.minerDamageRuleNaive( fatigueData )
    expectedRst = 2.5 
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleNaive_emptyInput_valueError():
    fatigueData = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleNaive( fatigueData )

def test_minerDamageRuleNaive_oneDimInput_valueError():
    fatigueData = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleNaive( fatigueData )

def test_minerDamageRuleNaive_irregularInput_valueError():
    fatigueData = [ [ -10, 100 ], [ 200, -2000 ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleNaive( fatigueData )

    fatigueData = [ [ 10, 100 ], [ 0, 0 ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleNaive( fatigueData )

    fatigueData = [ [ 10, 100 ], [ 2001, 2000 ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleNaive( fatigueData )

###############################################################################
# Test minerDamageRuleClassic
###############################################################################
def test_minerDamageRuleClassic_twoPairs_scalarOutput():
    lccData = [ [ 1, 100 ], [ 2, 10 ] ]
    snData = [ [ 10, 3 ], [ 1000, 1 ] ]
    fatigueLimit = 0.5
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.2 
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleClassic_threePairs_scalarOutput():
    lccData = [ [ 1, 1000 ], [ 2, 100 ], [ 4, 10 ] ]
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.12 
    np.testing.assert_allclose( calRst, expectedRst )

    lccData = [ [ 1, 1000 ], [ 3, 100 ], [ 4, 10 ] ]
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.21 
    np.testing.assert_allclose( calRst, expectedRst )

    lccData = [ [ 1, 1000 ], [ 3, 100 ], [ 4, 100 ] ]
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 1.11 
    np.testing.assert_allclose( calRst, expectedRst )

    lccData = [ [ 1, 1000 ], [ 3, 1000 ], [ 4, 100 ] ]
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 2.01 
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleClassic_threePairsHighFatigueLimit_scalarOutput():
    lccData = [ [ 1, 1000 ], [ 2, 100 ], [ 4, 10 ] ]
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 1
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.11 
    np.testing.assert_allclose( calRst, expectedRst )

    fatigueLimit = 2
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.1 
    np.testing.assert_allclose( calRst, expectedRst )

    fatigueLimit = 3
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.1 
    np.testing.assert_allclose( calRst, expectedRst )

    fatigueLimit = 4
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0
    np.testing.assert_allclose( calRst, expectedRst )

def test_minerDamageRuleClassic_emptyInput_valueError():
    lccData = [ [ ] ] 
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )

def test_minerDamageRuleClassic_oneDimInput_valueError():
    lccData = [ 1.0, 2.0, 3.0 ] 
    snData = [ [ 10, 5 ], [ 100, 4 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )

def test_minerDamageRuleClassic_irregularInput_valueError():
    lccData = [ [ 1, 1000 ], [ -2, 100 ], [ 4, 10 ] ]
    snData = [ [ 10, 5 ], [ 100000, 1 ] ]
    fatigueLimit = 0.5
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )

    lccData = [ [ 1, 1000 ], [ 2, -100 ], [ 4, 10 ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )

    lccData = [ [ 1, 1000 ], [ 0, 100 ], [ 4, 10 ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )

    lccData = [ [ 1, 1000 ], [ 2, 0 ], [ 4, 10 ] ]
    with pytest.raises( ValueError ):
        _ = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
