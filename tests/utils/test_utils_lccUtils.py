#!/usr/bin/env python3

from ffpack import utils
import numpy as np


###############################################################################
# Test cycleCountingAggregation
###############################################################################
def test_cycleCountingAggregation_onePairDefaultBinSize_oneCount():
    # case 1: closer to 0
    data = [ [ 0.2, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: closer to 1
    data = [ [ 0.7, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: in the middle
    data = [ [ 0.5, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 4: greater than 1 and closer to 2
    data = [ [ 1.7, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 5: greater than 1 and closer to 1
    data = [ [ 1.2, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 6: greater than 1 and in the middle
    data = [ [ 1.5, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_cycleCountingAggregation_twoPairsDefaultBinSize_countDepends():
    # case 1: aggregate to two bins
    data = [ [ 0.2, 2.0 ], [ 1.2, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 2.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: aggregate to one bin
    data = [ [ 0.7, 2.0 ], [ 1.2, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 3: aggregate to one bin - lager than binSize
    data = [ [ 1.7, 2.0 ], [ 2.2, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=1.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 1.0, 0.0 ], [ 2.0, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_cycleCountingAggregation_twoPairsSmallerBinSize_countDepends():
    # case 1: aggregate to two bins
    data = [ [ 0.3, 2.0 ], [ 0.9, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=0.5 )
    expectedRst = [ [ 0.0, 0.0 ], [ 0.5, 2.0 ], [ 1.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: aggregate to one bin
    data = [ [ 0.3, 2.0 ], [ 0.7, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=0.5 )
    expectedRst = [ [ 0.0, 0.0 ], [ 0.5, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )


def test_cycleCountingAggregation_twoPairsLargerBinSize_countDepends():
    # case 1: aggregate to two bins
    data = [ [ 0.3, 2.0 ], [ 2.9, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=2.0 )
    expectedRst = [ [ 0.0, 2.0 ], [ 2.0, 2.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )

    # case 2: aggregate to one bin
    data = [ [ 1.8, 2.0 ], [ 2.7, 2.0 ] ]
    calRst = utils.cycleCountingAggregation( data, binSize=2.0 )
    expectedRst = [ [ 0.0, 0.0 ], [ 2.0, 4.0 ] ]
    np.testing.assert_allclose( calRst, expectedRst )
