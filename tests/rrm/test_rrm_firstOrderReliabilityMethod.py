#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
import scipy as sp
import pytest
from unittest.mock import patch


###############################################################################
# Test form
###############################################################################
def test_form_twoNormalLinearCase_scalar():
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ]
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.5, 0.5 ]
    expectedXCoord = [ 0.5, 0.5 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_twoNormalNonLinearCase1_scalar():
    dim = 2

    def g( X ):
        return -1 * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 20

    dg = [ lambda X: -1, lambda X: -4 * X[ 1 ] ]
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 3.152380053229633
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.24999987, 3.14245128 ]
    expectedXCoord = [ 0.24999987, 3.14245128 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_twoNormalNonLinearCase2_scalar():
    dim = 2

    def g( X ):
        return -2 * X[ 0 ] * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 10

    dg = [ lambda X: -4 * X[ 0 ], lambda X: -4 * X[ 1 ] ]
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 2.23606797749979
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 1.58113883, 1.58113883 ]
    expectedXCoord = [ 1.58113883, 1.58113883 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_threeExpLinearCase_scalar():
    dim = 3

    def g( X ):
        return -np.sum( X ) + 3

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1 ]
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.expon()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.5845237835400737
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.33748047, 0.33748047, 0.33748047 ]
    expectedXCoord = [ 1.00000564, 1.00000564, 1.00000564 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_threeExpNonLinearCase_scalar():
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -2 * X[ 1 ], lambda X: -2 * X[ 2 ] ]
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.expon()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.5576668962820067
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.1549542, 0.37880177, 0.37880177 ]
    expectedXCoord = [ 0.82455805, 1.04293863, 1.04293863 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_twoExpOneNormalLinearCase_scalar():
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1 ]
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.norm()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.9406456373861823
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.57235988, 0.57235988, 0.47918946 ]
    expectedXCoord = [ 1.26040527, 1.26040527, 0.47918946 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_twoExpOneNormalNonLinearCase_scalar():
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -2 * X[ 1 ], lambda X: -2 * X[ 2 ] ]
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.norm()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.758619677951736
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 1.73566795e-01, 7.38497382e-01, 6.22122748e-06 ]
    expectedXCoord = [ 8.41408320e-01, 1.46921465e+00, 6.22122748e-06 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_twoExpOneNormalOneGammaLinearCase_scalar():
    dim = 4

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] + 4

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1, lambda X: -1 ]
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.norm()
    X4 = sp.stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.44811639358366484
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.18690136, 0.18690136, 0.20303017, 0.29953768 ]
    expectedXCoord = [ 0.85362352, 0.85362352, 0.20303017, 2.08972279 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_form_twoExpOneNormalOneGammaNonLinearCase_scalar():
    dim = 4

    def g( X ):
        return -X[ 0 ] * X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] * X[ 3 ] + 6

    dg = [ lambda X: -1, lambda X: -2 * X[ 0 ], lambda X: -1, lambda X: -2 * X[ 3 ] ]
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.norm()
    X4 = sp.stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.359925328833402
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.04399299, 0.06519399, 0.05325441, 0.34716401 ]
    expectedXCoord = [ 0.72886764, 0.74652734, 0.05325441, 2.16077627 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )
