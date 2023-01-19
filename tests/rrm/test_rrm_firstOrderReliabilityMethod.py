#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
from scipy import stats
import pytest
from unittest.mock import patch
from ffpack.rpm import NatafTransformation


###############################################################################
# Test formHLRF
###############################################################################
def test_formHLRF_dimLessOneCase_valueError( ):
    dim = 0

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )


def test_formHLRF_distObjsDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    distObjs = [ X1 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )


def test_formHLRF_corrMatDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0 ], [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )
    

def test_formHLRF_corrMatNotTwoDimCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )


def test_formHLRF_corrMatNotSymmCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.5 ], [ -0.5, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )


def test_formHLRF_corrMatNotPositiveDefiniteCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -1.2 ], [ -1.2, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )


def test_formHLRF_corrMatDiagNotOneCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat )


def test_formHLRF_notConverge_valueError():
    dim = 2

    def g( X ):
        return X[ 0 ] ** 4 + 2 * X[ 1 ] ** 4 - 20

    dg = [ lambda X: 4 * X[ 0 ] ** 3, lambda X: 8 * X[ 1 ] ** 3 ] 
    X1 = stats.norm( loc=10.0, scale=5.0 )
    X2 = stats.norm( loc=10.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formHLRF( dim, g, dg, distObjs, corrMat, iter=100 )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
@patch.object( NatafTransformation, 'getX' )
def test_formHLRF_twoNormalLinearMockCase1_scalar( mock_getX, dgExists ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_getX.return_value = ( np.array( [ 1.0, 1.0 ] ), 
                               np.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] ) )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, distObjs, 
                                                         corrMat, iter=1 )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.5, 0.5 ]
    expectedXCoord = [ 1.0, 1.0 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
@patch.object( NatafTransformation, 'getX' )
def test_formHLRF_twoNormalLinearMockCase2_scalar( mock_getX, dgExists ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_getX.return_value = ( np.array( [ 2.0, 2.0 ] ), 
                               np.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] ) )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, distObjs, 
                                                         corrMat, iter=1 )
    expectedBeta = -1 * np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ -0.5, -0.5 ]
    expectedXCoord = [ 2.0, 2.0 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
@patch.object( NatafTransformation, 'getX' )
def test_formHLRF_twoNormalNonLinearMockCase_scalar( mock_getX, dgExists ):
    dim = 2

    def g( X ):
        return -1 * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 20

    dg = [ lambda X: -1, lambda X: -4 * X[ 1 ] ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_getX.return_value = ( np.array( [ 1.0, 1.0 ] ), 
                               np.array( [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ] ) )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, distObjs, 
                                                         corrMat, iter=1 )
    expectedBeta = 22 / np.sqrt( 17 )
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 22 / 17, 88 / 17 ]
    expectedXCoord = [ 1.0, 1.0 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoNormalLinearCase_scalar( dgExists ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.5, 0.5 ]
    expectedXCoord = [ 0.5, 0.5 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoNormalNonLinearCase1_scalar( dgExists ):
    dim = 2

    def g( X ):
        return -1 * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 20

    dg = [ lambda X: -1, lambda X: -4 * X[ 1 ] ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 3.152380053229633
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.24999987, 3.14245128 ]
    expectedXCoord = [ 0.24999987, 3.14245128 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoNormalNonLinearCase2_scalar( dgExists ):
    dim = 2

    def g( X ):
        return -2 * X[ 0 ] * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 10

    dg = [ lambda X: -4 * X[ 0 ], lambda X: -4 * X[ 1 ] ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 2.23606797749979
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 1.58113883, 1.58113883 ]
    expectedXCoord = [ 1.58113883, 1.58113883 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoNormalNonLinearCase3_scalar( dgExists ):
    dim = 2

    def g( X ):
        return X[ 0 ] ** 2 + 2 * X[ 1 ] ** 2 - 20

    dg = [ lambda X: 2 * X[ 0 ], lambda X: 4 * X[ 1 ] ] if dgExists else None
    X1 = stats.norm( loc=5.0, scale=5.0 )
    X2 = stats.norm( loc=5.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.6636720072645971
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ -0.37621928, -0.54673539 ]
    expectedXCoord = [ 3.11890359, 2.26632306 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoNormalNonLinearCase4_scalar( dgExists ):
    dim = 2

    def g( X ):
        return X[ 0 ] ** 2 + 2 * X[ 1 ] ** 2 - 20

    dg = [ lambda X: 2 * X[ 0 ], lambda X: 4 * X[ 1 ] ] if dgExists else None
    X1 = stats.norm( loc=3.0, scale=6.0 )
    X2 = stats.norm( loc=4.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.28257129279933346
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ -0.12510462, -0.25336805 ]
    expectedXCoord = [ 2.2493723, 2.73315974 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_threeExpLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -np.sum( X ) + 3

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.expon()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.5845237835400737
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.33748047, 0.33748047, 0.33748047 ]
    expectedXCoord = [ 1.00000564, 1.00000564, 1.00000564 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_threeExpNonLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -2 * X[ 1 ], lambda X: -2 * X[ 2 ] ] \
        if dgExists else None
    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.expon()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.5576668962820067
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.1549542, 0.37880177, 0.37880177 ]
    expectedXCoord = [ 0.82455805, 1.04293863, 1.04293863 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoExpOneNormalLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.9406456373861823
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.57235988, 0.57235988, 0.47918946 ]
    expectedXCoord = [ 1.26040527, 1.26040527, 0.47918946 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoExpOneNormalNonLinearCase_scalar( dgExists ):
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    dg = [ lambda X: -1, lambda X: -2 * X[ 1 ], lambda X: -2 * X[ 2 ] ] \
        if dgExists else None
    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.758619677951736
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 1.73566795e-01, 7.38497382e-01, 6.22122748e-06 ]
    expectedXCoord = [ 8.41408320e-01, 1.46921465e+00, 6.22122748e-06 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoExpOneNormalOneGammaLinearCase_scalar( dgExists ):
    dim = 4

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] + 4

    dg = [ lambda X: -1, lambda X: -1, lambda X: -1, lambda X: -1 ] \
        if dgExists else None
    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    X4 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.44811639358366484
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.18690136, 0.18690136, 0.20303017, 0.29953768 ]
    expectedXCoord = [ 0.85362352, 0.85362352, 0.20303017, 2.08972279 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_formHLRF_twoExpOneNormalOneGammaNonLinearCase_scalar( dgExists ):
    dim = 4

    def g( X ):
        return -X[ 0 ] * X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] * X[ 3 ] + 6

    dg = [ lambda X: -2 * X[ 0 ],
           lambda X: -1, 
           lambda X: -1, 
           lambda X: -2 * X[ 3 ] ] if dgExists else None
    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    X4 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formHLRF( dim, g, dg, 
                                                         distObjs, corrMat )
    expectedBeta = 0.35867913670082807
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.06679387, 0.04387002, 0.05311046, 0.34560672 ]
    expectedXCoord = [ 0.74787182, 0.72876606, 0.05311046, 2.15842795 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )



###############################################################################
# Test formCOPT
###############################################################################
def test_formCOPT_dimLessOneCase_valueError( ):
    dim = 0

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )


def test_formCOPT_distObjsDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    distObjs = [ X1 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )


def test_formCOPT_corrMatDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0 ], [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )
    

def test_formCOPT_corrMatNotTwoDimCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )


def test_formCOPT_corrMatNotSymmCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.5 ], [ -0.5, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )


def test_formCOPT_corrMatNotPositiveDefiniteCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -1.2 ], [ -1.2, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )


def test_formCOPT_corrMatDiagNotOneCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.formCOPT( dim, g, distObjs, corrMat )


def test_formCOPT_twoNormalLinearCase_scalar( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.5, 0.5 ]
    expectedXCoord = [ 0.5, 0.5 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoNormalNonLinearCase1_scalar():
    dim = 2

    def g( X ):
        return -1 * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 20

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 3.152380053229633
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.24999987, 3.14245128 ]
    expectedXCoord = [ 0.24999987, 3.14245128 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoNormalNonLinearCase2_scalar():
    dim = 2

    def g( X ):
        return -2 * X[ 0 ] * X[ 0 ] - 2 * X[ 1 ] * X[ 1 ] + 10

    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 2.23606797749979
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 1.58113883, 1.58113883 ]
    expectedXCoord = [ 1.58113883, 1.58113883 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoNormalNonLinearCase3_scalar():
    dim = 2

    def g( X ):
        return X[ 0 ] ** 2 + 2 * X[ 1 ] ** 2 - 20

    X1 = stats.norm( loc=5.0, scale=5.0 )
    X2 = stats.norm( loc=5.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.6636720072645971
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ -0.37621928, -0.54673539 ]
    expectedXCoord = [ 3.11890359, 2.26632306 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


def test_formCOPT_twoNormalNonLinearCase4_scalar():
    dim = 2

    def g( X ):
        return X[ 0 ] ** 2 + 2 * X[ 1 ] ** 2 - 20

    X1 = stats.norm( loc=3.0, scale=6.0 )
    X2 = stats.norm( loc=4.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.28257129279933346
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ -0.12510462, -0.25336805 ]
    expectedXCoord = [ 2.2493723, 2.73315974 ]
    np.testing.assert_allclose( np.round( expectedBeta, 5 ), np.round( calBeta, 5 ) )
    np.testing.assert_allclose( np.round( expectedPf, 5 ), np.round( calPf, 5 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_threeExpLinearCase_scalar():
    dim = 3

    def g( X ):
        return -np.sum( X ) + 3

    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.expon()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.5845237835400737
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.33748047, 0.33748047, 0.33748047 ]
    expectedXCoord = [ 1.00000564, 1.00000564, 1.00000564 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_threeExpNonLinearCase_scalar():
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.expon()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.5576668962820067
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.1549542, 0.37880177, 0.37880177 ]
    expectedXCoord = [ 0.82455805, 1.04293863, 1.04293863 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoExpOneNormalLinearCase_scalar():
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] + 3

    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.9406456373861823
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.57235988, 0.57235988, 0.47918946 ]
    expectedXCoord = [ 1.26040527, 1.26040527, 0.47918946 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoExpOneNormalNonLinearCase_scalar():
    dim = 3

    def g( X ):
        return -X[ 0 ] - X[ 1 ] * X[ 1 ] - X[ 2 ] * X[ 2 ] + 3

    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    distObjs = [ X1, X2, X3 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.758619677951736
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 1.73566795e-01, 7.38497382e-01, 6.22122748e-06 ]
    expectedXCoord = [ 8.41408320e-01, 1.46921465e+00, 6.22122748e-06 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoExpOneNormalOneGammaLinearCase_scalar():
    dim = 4

    def g( X ):
        return -X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] + 6

    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    X4 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 1.2634328903061505
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.54715364, 0.54715364, 0.46536261, 0.88371154 ]
    expectedXCoord = [ 1.23053381, 1.23053381, 0.46536261, 3.07356976 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )


def test_formCOPT_twoExpOneNormalOneGammaNonLinearCase_scalar():
    dim = 4

    def g( X ):
        return -X[ 0 ] * X[ 0 ] - X[ 1 ] - X[ 2 ] - X[ 3 ] * X[ 3 ] + 6

    X1 = stats.expon()
    X2 = stats.expon()
    X3 = stats.norm()
    X4 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3, X4 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.formCOPT( dim, g, distObjs, corrMat )
    expectedBeta = 0.35867913670082807
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.06679387, 0.04387002, 0.05311046, 0.34560672 ]
    expectedXCoord = [ 0.74787182, 0.72876606, 0.05311046, 2.15842795 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 3 ), 
                                np.round( calUCoord, 3 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 3 ), 
                                np.round( calXCoord, 3 ) )
