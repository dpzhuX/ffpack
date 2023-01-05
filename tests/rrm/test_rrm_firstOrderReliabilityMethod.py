#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
import scipy as sp
import pytest
from unittest.mock import Mock
from ffpack.rpm import NatafTransformation


###############################################################################
# Test form
###############################################################################
def test_form_dimLessOneCase_valueError( ):
    dim = 0

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )


def test_form_distObjsDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    distObjs = [ X1 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )


def test_form_corrMatDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0 ], [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )
    

def test_form_corrMatNotTwoDimCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )


def test_form_corrMatNotSymmCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.5 ], [ -0.5, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )


def test_form_corrMatNotPositiveDefiniteCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -1.2 ], [ -1.2, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )


def test_form_corrMatDiagNotOneCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = sp.stats.norm()
    X2 = sp.stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.form( dim, g, dg, distObjs, corrMat )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoNormalLinearCase_scalar( dgExists ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] if dgExists else None
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


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoNormalNonLinearCase1_scalar( dgExists ):
    dim = 2

    def g( X ):
        return -1 * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 20

    dg = [ lambda X: -1, lambda X: -4 * X[ 1 ] ] if dgExists else None
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


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoNormalNonLinearCase2_scalar( dgExists ):
    dim = 2

    def g( X ):
        return -2 * X[ 0 ] * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 10

    dg = [ lambda X: -4 * X[ 0 ], lambda X: -4 * X[ 1 ] ] if dgExists else None
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


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_threeExpLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -np.sum( X ) + 3

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1 ] if dgExists else None
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


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_threeExpNonLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -2 * X[ 1 ], lambda X: -2 * X[ 2 ] ] \
        if dgExists else None
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


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoExpOneNormalLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1 ] if dgExists else None
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
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoExpOneNormalNonLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -2 * X[ 1 ], lambda X: -2 * X[ 2 ] ] \
        if dgExists else None
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
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoExpOneNormalOneGammaLinearCase_scalar( dgExists ):
    dim = 4

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] + 4

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1, lambda X: -1 ] \
        if dgExists else None
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
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_form_twoExpOneNormalOneGammaNonLinearCase_scalar( dgExists ):
    dim = 4

    def g( X ):
        return -X[ 0 ] * X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] * X[ 3 ] + 6

    dg = [ lambda X: -2 * X[ 0 ],
           lambda X: -1, 
           lambda X: -1, 
           lambda X: -2 * X[ 3 ] ] if dgExists else None
    X1 = sp.stats.expon()
    X2 = sp.stats.expon()
    X3 = sp.stats.norm()
    X4 = sp.stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.form( dim, g, dg, distObjs, corrMat )
    expectedBeta = 0.35867913670082807
    expectedPf = sp.stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.06679387, 0.04387002, 0.05311046, 0.34560672 ]
    expectedXCoord = [ 0.74787182, 0.72876606, 0.05311046, 2.15842795 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )
