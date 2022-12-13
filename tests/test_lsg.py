#!/usr/bin/env python3

from ffpack import lsg
import numpy as np
import pytest


###############################################################################
# Test randomWalk
###############################################################################
def test_randomWalk_normalUseCase_pass():
    print(lsg.randomWalk( 5, 2 ))

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