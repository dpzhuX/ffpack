#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
from scipy import stats
import pytest
from unittest.mock import patch


###############################################################################
# Test subsetSimulation
###############################################################################
def test_subsetSimulation_normalCase_scalar():
    dim = 2

    def g( X ):
        return 4 - X[ 0 ] / 4 + np.sin( 5 * X[ 0 ] ) - X[ 1 ]

    # def g( X ):
    #     return -np.sum( X ) + 1

    # def g( X ):
    #     return -1 * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 20

    # def g( X ):
    #     return X[ 0 ] ** 4 + 2 * X[ 1 ] ** 4 - 20

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    numSamples, numSubsets = 500, 10
    np.random.seed(0)
    rrm.subsetSimulation( dim, g, distObjs, corrMat, numSamples, numSubsets )


test_subsetSimulation_normalCase_scalar()
