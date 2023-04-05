#!/usr/bin/env python3

from ffpack import lcc
import numpy as np
import pytest


##############################################################################
# Test goodmanCorrection function
###############################################################################
def test_goodmanCorrection_incorrectStressRange_valueError():
    ultimateStrength = 2.0
    stressRange = 1
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )

    stressRange = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )

    stressRange = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )
    
    stressRange = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )

    stressRange = [ 1.0, 2.0, 3.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )

    stressRange = [ 1.0, 0.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )

    stressRange = [ -2.0, -1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )


def test_goodmanCorrection_incorrectUltimateStrength_valueError():
    data = [ 1.0, 2.0 ]

    ultimateStrength = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, ultimateStrength )

    ultimateStrength = -1
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, ultimateStrength )
        
    ultimateStrength = 0
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, ultimateStrength )

    ultimateStrength = 1.4
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( data, ultimateStrength )


def test_goodmanCorrection_stressRatioOutOfBound_valueError():
    ultimateStrength = 2.0

    # case 1: stress ratio R = -inf
    stressRange = [ -1.0, 0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )

    # case 2: R < -1
    stressRange = [ -2.0, 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength )  


def test_goodmanCorrection_invalidN_valueError():
    stressRange = [ 1.0, 2.0 ]
    ultimateStrength = 2.0

    n = 0.0
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength, n )


def test_goodmanCorrection_incorrectStrengthWithN_valueError():
    stressRange = [ 1.0, 3.0 ]
    ultimateStrength = 3.0

    n = 2.0
    with pytest.raises( ValueError ):
        _ = lcc.goodmanCorrection( stressRange, ultimateStrength, n )


def test_goodmanCorrection_normalCase_pass():
    # case 0: normal, all positive
    stressRange = [ 1.0, 2.0 ]
    ultimateStrength = 2.0
    calRst = lcc.goodmanCorrection( stressRange, ultimateStrength )
    expectedRst = 2.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 1: stress ratio R = -1 -> zero mean
    stressRange = [ -1.0, 1.0 ]
    ultimateStrength = 2.0
    calRst = lcc.goodmanCorrection( stressRange, ultimateStrength )
    expectedRst = 1.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: R = 0 -> lower stress = 0
    stressRange = [ 0, 2.0 ]
    ultimateStrength = 2.0
    calRst = lcc.goodmanCorrection( stressRange, ultimateStrength )
    expectedRst = 2.0
    np.testing.assert_allclose( calRst, expectedRst )


##############################################################################
# Test soderbergCorrection function
###############################################################################
def test_soderbergCorrection_incorrectStressRange_valueError():
    yieldStrength = 2.0
    stressRange = 1
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )

    stressRange = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )

    stressRange = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )
    
    stressRange = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )

    stressRange = [ 1.0, 2.0, 3.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )

    stressRange = [ 1.0, 0.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )

    stressRange = [ -2.0, -1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )


def test_soderbergCorrectionn_incorrectYieldStrength_valueError():
    data = [ 1.0, 2.0 ]

    yieldStrength = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( data, yieldStrength )

    yieldStrength = -1
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( data, yieldStrength )
        
    yieldStrength = 0
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( data, yieldStrength )

    yieldStrength = 1.4
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( data, yieldStrength )


def test_soderbergCorrection_stressRatioOutOfBound_valueError():
    yieldStrength = 2.0

    # case 1: stress ratio R = -inf
    stressRange = [ -1.0, 0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )

    # case 2: R < -1
    stressRange = [ -2.0, 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength )  


def test_soderbergCorrection_invalidN_valueError():
    stressRange = [ 1.0, 2.0 ]
    yieldStrength = 2.0

    n = 0.0
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, yieldStrength, n )


def test_soderbergCorrection_incorrectStrengthWithN_valueError():
    stressRange = [ 1.0, 3.0 ]
    ultimateStrength = 3.0

    n = 2.0
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, ultimateStrength, n )


def test_soderbergCorrection_normalCase_pass():
    # case 0: normal, all positive
    stressRange = [ 1.0, 2.0 ]
    yieldStrength = 2.0
    calRst = lcc.soderbergCorrection( stressRange, yieldStrength )
    expectedRst = 2.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 1: stress ratio R = -1 -> zero mean
    stressRange = [ -1.0, 1.0 ]
    yieldStrength = 2.0
    calRst = lcc.soderbergCorrection( stressRange, yieldStrength )
    expectedRst = 1.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: R = 0 -> lower stress = 0
    stressRange = [ 0, 2.0 ]
    yieldStrength = 2.0
    calRst = lcc.soderbergCorrection( stressRange, yieldStrength )
    expectedRst = 2.0
    np.testing.assert_allclose( calRst, expectedRst )


##############################################################################
# Test gerberCorrection function
###############################################################################
def test_gerberCorrection_incorrectStressRange_valueError():
    ultimateStrength = 2.0
    stressRange = 1
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )

    stressRange = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )

    stressRange = [ [ ] ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )
    
    stressRange = [ 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )

    stressRange = [ 1.0, 2.0, 3.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )

    stressRange = [ 1.0, 0.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )

    stressRange = [ -2.0, -1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength )


def test_gerberCorrection_incorrectUltimateStrength_valueError():
    data = [ 1.0, 2.0 ]

    ultimateStrength = [ ]
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( data, ultimateStrength )

    ultimateStrength = -1
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( data, ultimateStrength )
        
    ultimateStrength = 0
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( data, ultimateStrength )

    ultimateStrength = 1.4
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( data, ultimateStrength )


def test_gerberCorrection_stressRatioOutOfBound_valueError():
    ultimateStrength = 2.0

    # case 1: stress ratio R = -inf
    stressRange = [ -1.0, 0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, ultimateStrength )

    # case 2: R < -1
    stressRange = [ -2.0, 1.0 ]
    with pytest.raises( ValueError ):
        _ = lcc.soderbergCorrection( stressRange, ultimateStrength )  


def test_gerberCorrection_invalidN_valueError():
    stressRange = [ 1.0, 2.0 ]
    ultimateStrength = 2.0

    n = 0.0
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength, n )


def test_gerberCorrection_incorrectStrengthWithN_valueError():
    stressRange = [ 1.0, 3.0 ]
    ultimateStrength = 3.0

    n = 2.0
    with pytest.raises( ValueError ):
        _ = lcc.gerberCorrection( stressRange, ultimateStrength, n )
    

def test_gerberCorrection_normalCase_pass():
    # case 0: normal, all positive
    stressRange = [ 1.0, 2.0 ]
    ultimateStrength = 4.5
    calRst = lcc.gerberCorrection( stressRange, ultimateStrength )
    expectedRst = 0.5625
    np.testing.assert_allclose( calRst, expectedRst )

    # case 1: stress ratio R = -1 -> zero mean
    stressRange = [ -1.0, 1.0 ]
    ultimateStrength = 2.0
    calRst = lcc.gerberCorrection( stressRange, ultimateStrength )
    expectedRst = 1.0
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: R = 0 -> lower stress = 0
    stressRange = [ 0, 2.0 ]
    ultimateStrength = 3.0
    calRst = lcc.gerberCorrection( stressRange, ultimateStrength )
    expectedRst = 1.125
    np.testing.assert_allclose( calRst, expectedRst )
