#!/usr/bin/env python3

from ffpack import lsm
import numpy as np
import pytest


###############################################################################
# Test randomWalkUniform
###############################################################################
def test_astmSimpleRangeCountingMatrix_severalPoints_2dMatrix():
    data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    calRst = lsm.astmSimpleRangeCountingMatrix( data, method="minMax", digitization=True, resolution=1.0 )
    print( calRst )

