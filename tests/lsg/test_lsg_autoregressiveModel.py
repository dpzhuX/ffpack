#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest


###############################################################################
# Test arNormal
###############################################################################
def test_arNormal_numStepsLessThanOneCase_valueError():
    obs = [ 0, 0  ]
    phis = [ 0.5, 0.3 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( -1, obs, phis, 0, 0.5 )

    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 0, obs, phis, 0, 0.5 )


def test_arNormal_lengthsNotEqualCase_valueError():
    obs = [ 0, 0  ]
    phis = [ 0.3 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )

    obs = [ 0 ]
    phis = [ 0.3, 0.5 ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )


def test_arNormal_enptyObsAndPhisCase_valueError():
    obs = [ ]
    phis = [ ]
    with pytest.raises( ValueError ):
        _ = lsg.arNormal( 500, obs, phis, 0, 0.5 )
