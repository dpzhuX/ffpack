#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
from scipy import stats
import pytest
from unittest.mock import patch


###############################################################################
# Test mainCurvaturesAtDesignPoint
###############################################################################
def test_mainCurvaturesAtDesignPoint_dimLessOneCase_valueError( ):
    dim = 0

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )


def test_mainCurvaturesAtDesignPoint_distObjsDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    distObjs = [ X1 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )


def test_mainCurvaturesAtDesignPoint_corrMatDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0 ], [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )
    

def test_mainCurvaturesAtDesignPoint_corrMatNotTwoDimCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )


def test_mainCurvaturesAtDesignPoint_corrMatNotSymmCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.5 ], [ -0.5, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )


def test_mainCurvaturesAtDesignPoint_corrMatNotPositiveDefiniteCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -1.2 ], [ -1.2, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )


def test_mainCurvaturesAtDesignPoint_corrMatDiagNotOneCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_mainCurvaturesAtDesignPoint_twoNormalLinearCase_scalar( dgExists ):
    # Linear case should have the same results as FORM.
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calKs, calBeta, calUCoord, calXCoord = rrm.\
        mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )
    expectedKs = [ 0.0 ]
    expectedBeta = np.sqrt( 2 ) / 2
    expectedUCoord = [ 0.5, 0.5 ]
    expectedXCoord = [ 0.5, 0.5 ]
    np.testing.assert_allclose( expectedKs, calKs, atol=1e-4 )
    np.testing.assert_allclose( expectedBeta, calBeta, atol=1e-4 )
    np.testing.assert_allclose( expectedUCoord, calUCoord, atol=1e-4 )
    np.testing.assert_allclose( expectedXCoord, calXCoord, atol=1e-4 )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_mainCurvaturesAtDesignPoint_twoNormalNonLinearCase_scalar( dgExists ):
    r'''
    This example comes from the reference [Choi2007] page 132.

    Here are some expected values of local variables from the refernce:

    lsfGradAtU = [ 119.8184, 124.8218 ]
    lsfHmAtU = [ [ 989.5592, 0 ],
                 [ 0, 1281.2632 ] ]
    HBH = [ [ 6.5278, -0.8423 ],
            [ -0.8423, 6.5968 ] ]

    Notes
    -----
    formHLRF does not converge for this high order limit state function.

    References
    ----------
    .. [Choi2007] Choi, S.K., Canfield, R.A. and Grandhi, R.V., 2007. 
       Reliability-based Structural Design. Springer London.
    '''
    dim = 2

    def g( X ):
        return X[ 0 ] ** 4 + 2 * X[ 1 ] ** 4 - 20

    dg = [ lambda X: 4 * X[ 0 ] ** 3, lambda X: 8 * X[ 1 ] ** 3 ] if dgExists \
        else None
    X1 = stats.norm( loc=10.0, scale=5.0 )
    X2 = stats.norm( loc=10.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calKs, calBeta, calUCoord, calXCoord = rrm.\
        mainCurvaturesAtDesignPoint( dim, g, dg, distObjs, corrMat )
    expectedKs = [ 6.5278 ]
    expectedBeta = 2.3654
    expectedUCoord = [ -1.6368, -1.7077 ]
    expectedXCoord = [ 1.8162, 1.4613 ]
    np.testing.assert_allclose( expectedKs, calKs, atol=1e-2 )
    np.testing.assert_allclose( expectedBeta, calBeta, atol=1e-4 )
    np.testing.assert_allclose( expectedUCoord, calUCoord, atol=1e-3 )
    np.testing.assert_allclose( expectedXCoord, calXCoord, atol=1e-3 )


###############################################################################
# Test sormBreitung
###############################################################################
@patch( "ffpack.rrm.secondOrderReliabilityMethod.mainCurvaturesAtDesignPoint" )
def test_sormBreitung_twoNormalLinearCase_scalar( mock_curv ):
    # Linear case should have the same results as FORM.
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ]
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_curv.return_value = [ [ 0.0 ], np.sqrt( 2 ) / 2, 
                               [ 0.5, 0.5 ], [ 0.5, 0.5 ] ]
    _, calPf, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )


@patch( "ffpack.rrm.secondOrderReliabilityMethod.mainCurvaturesAtDesignPoint" )
def test_sormBreitung_twoNormalNonLinearCase_scalar( mock_curv ):
    r'''
    This example comes from the reference [Choi2007] page 132.

    References
    ----------
    .. [Choi2007] Choi, S.K., Canfield, R.A. and Grandhi, R.V., 2007. 
       Reliability-based Structural Design. Springer London.
    '''
    dim = 2

    def g( X ):
        return X[ 0 ] ** 4 + 2 * X[ 1 ] ** 4 - 20

    dg = [ lambda X: 4 * X[ 0 ] ** 3, lambda X: 8 * X[ 1 ] ** 3 ]
    X1 = stats.norm( loc=10.0, scale=5.0 )
    X2 = stats.norm( loc=10.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_curv.return_value = [ [ 6.5278 ], 2.3654,
                               [ -1.6368, -1.7077 ], [ 1.8162, 1.4613 ] ]
    _, calPf, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )
    expectedPf = 0.00222059
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )


###############################################################################
# Test sormTvedt
###############################################################################
@patch( "ffpack.rrm.secondOrderReliabilityMethod.mainCurvaturesAtDesignPoint" )
def test_sormTvedt_twoNormalLinearCase_scalar( mock_curv ):
    # Linear case should have the same results as FORM.
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ]
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_curv.return_value = [ [ 0.0 ], np.sqrt( 2 ) / 2, 
                               [ 0.5, 0.5 ], [ 0.5, 0.5 ] ]
    _, calPf, _, _ = rrm.sormTvedt( dim, g, dg, distObjs, corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )


@patch( "ffpack.rrm.secondOrderReliabilityMethod.mainCurvaturesAtDesignPoint" )
def test_sormTvedt_twoNormalNonLinearCase_scalar( mock_curv ):
    r'''
    This example comes from the reference [Choi2007] page 135.

    There is some errors in the reference. Here should be the correct value:
    k = 6.5278
    beta = 2.3654
    cdf( -beta ) = 0.00900530
    pdf( beta ) = 0.02431901


    beta * cdf( -beta ) - pdf( beta ) 
        = 2.3654 * 0.00900530 - 0.02431901
        = -0.00301787338
    ( 1 + beta * k ) ** -0.5 
        = ( 1 + 2.3654 * 6.5278 ) ** -0.5
        = 0.2466253753552
    ( 1 + ( beta + 1 ) * k ) ** -0.5 
        = ( 1 + 3.3654 * 6.5278 ) ** -0.5
        = 0.20865662972

    A1 = 0.00900530 * 0.2466253753552 = 0.0022209354927
    A2 = ( -0.00301787338 ) * ( 0.2466253753552 - 0.20865662972 ) = -0.000114584867
    A3 = 3.3654 * ( -0.00301787338 ) * ( 0.2466253753552 - 0.20865662972 )
       = -0.0003856239105

    References
    ----------
    .. [Choi2007] Choi, S.K., Canfield, R.A. and Grandhi, R.V., 2007. 
       Reliability-based Structural Design. Springer London.
    '''
    dim = 2

    def g( X ):
        return X[ 0 ] ** 4 + 2 * X[ 1 ] ** 4 - 20

    dg = [ lambda X: 4 * X[ 0 ] ** 3, lambda X: 8 * X[ 1 ] ** 3 ] 
    X1 = stats.norm( loc=10.0, scale=5.0 )
    X2 = stats.norm( loc=10.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_curv.return_value = [ [ 6.5278 ], 2.3654,
                               [ -1.6368, -1.7077 ], [ 1.8162, 1.4613 ] ]
    _, calPf, _, _ = rrm.sormTvedt( dim, g, dg, distObjs, corrMat )
    expectedPf = 0.0022209354927 + ( -0.000114584867 ) + ( -0.0003856239105 )
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )


###############################################################################
# Test sormHRack
###############################################################################
@patch( "ffpack.rrm.secondOrderReliabilityMethod.mainCurvaturesAtDesignPoint" )
def test_sormHRack_twoNormalLinearCase_scalar( mock_curv ):
    # Linear case should have the same results as FORM.
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ]
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_curv.return_value = [ [ 0.0 ], np.sqrt( 2 ) / 2, 
                               [ 0.5, 0.5 ], [ 0.5, 0.5 ] ]
    _, calPf, _, _ = rrm.sormHRack( dim, g, dg, distObjs, corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )


@patch( "ffpack.rrm.secondOrderReliabilityMethod.mainCurvaturesAtDesignPoint" )
def test_sormHRack_twoNormalNonLinearCase_scalar( mock_curv ):
    r'''
    This data comes from the reference [Choi2007] page 135.

    There is some errors in the reference. Here should be the correct value:
    k = 6.5278
    beta = 2.3654
    cdf( beta ) = 0.99099470
    cdf( -beta ) = 0.00900530
    pdf( beta ) = 0.02431901


    ( 1 + k * pdf( beta ) / cdf( beta ) ) ** -0.5 
        = ( 1 + 6.5278 * 0.02431901 / 0.99099470 ) ** -0.5
        = 0.928399776
    
    pf = 0.00900530 * 0.928399776 = 0.0083605185


    References
    ----------
    .. [Choi2007] Choi, S.K., Canfield, R.A. and Grandhi, R.V., 2007. 
       Reliability-based Structural Design. Springer London.
    '''
    dim = 2

    def g( X ):
        return X[ 0 ] ** 4 + 2 * X[ 1 ] ** 4 - 20

    dg = [ lambda X: 4 * X[ 0 ] ** 3, lambda X: 8 * X[ 1 ] ** 3 ] 
    X1 = stats.norm( loc=10.0, scale=5.0 )
    X2 = stats.norm( loc=10.0, scale=5.0 )
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    mock_curv.return_value = [ [ 6.5278 ], 2.3654,
                               [ -1.6368, -1.7077 ], [ 1.8162, 1.4613 ] ]
    _, calPf, _, _ = rrm.sormHRack( dim, g, dg, distObjs, corrMat )
    expectedPf = 0.0083605185
    np.testing.assert_allclose( expectedPf, calPf, atol=1e-6 )
