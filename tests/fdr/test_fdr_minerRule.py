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

###############################################################################
# Test minerDamageRuleClassic
###############################################################################
def test_minerDamageRuleNaive_twoPairs_scalarOutput():
    lccData = [ [ 1, 100 ], [ 2, 10 ] ]
    snData = [ [ 10, 3 ], [ 1000, 1 ] ]
    fatigueLimit = 0.5
    calRst = fdr.minerDamageRuleClassic( lccData, snData, fatigueLimit )
    expectedRst = 0.2 
    np.testing.assert_allclose( calRst, expectedRst )
