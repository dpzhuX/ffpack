#!/usr/bin/env python3

from ffpack import lcc
import numpy as np
import pytest


##############################################################################
# Test goodmanCorrection function
###############################################################################
def test_goodmanCorrection_incorrectStressRange_valueError():
    sigma = 2.0
    stressRange = 1
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )

    stressRange = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )

    stressRange = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )
    
    stressRange = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )

    stressRange = [ 1.0, 2.0, 3.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )

    stressRange = [ 1.0, 0.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )

    stressRange = [ -2.0, -1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )


def test_goodmanCorrection_incorrectSigma_valueError():
    data = [ 1.0, 2.0 ]

    sigma = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, sigma )

    sigma = -1
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, sigma )
        
    sigma = 0
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, sigma )

    sigma = 1.8
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, sigma )


def test_goodmanCorrection_stressRatioOutOfBound_valueError():
    sigma = 2.0

    # case 1: stress ratio R = -inf
    stressRange = [ -1.0, 0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )

    # case 2: R < -1
    stressRange = [ -2.0, 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma )  


def test_goodmanCorrection_invalidN_valueError():
    stressRange = [ 1.0, 2.0 ]
    sigma = 2.0

    n = 0.0
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, sigma, n )


def test_goodmanCorrection_normalCase_pass():
    # case 0: normal, all positive
    stressRange = [ 1.0, 2.0 ]
    sigma = 2.0
    calRst = lcc.goodmanCorrection( stressRange, sigma )
    expectedRst = 2.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 1: stress ratio R = -1 -> zero mean
    stressRange = [ -1.0, 1.0 ]
    sigma = 2.0
    calRst = lcc.goodmanCorrection( stressRange, sigma )
    expectedRst = 1.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: R = 0 -> lower stress = 0
    stressRange = [ 0, 2.0 ]
    sigma = 2.0
    calRst = lcc.goodmanCorrection( stressRange, sigma )
    expectedRst = 2.0
    np.testing.assert_allclose( calRst, expectedRst )
