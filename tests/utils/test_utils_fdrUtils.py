
#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest

###############################################################################
# Test SnCurveFitter
###############################################################################
def test_snCurverFitter_twoPairsData_queryPass():
    data = [ [ 10, 4 ], [ 10000, 1 ] ]
    snCurveFitter = utils.SnCurveFitter( data, fatigueLimit=0.5 )
    assert snCurveFitter.getN( 4 ) == 10
    assert snCurveFitter.getN( 3 ) == 100
    assert snCurveFitter.getN( 2 ) == 1000
    assert snCurveFitter.getN( 1 ) == 10000
    assert snCurveFitter.getN( 0.5 ) == -1