#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
from scipy import stats
import pytest
from unittest.mock import patch
from ffpack.rrm import formHLRF
from ffpack.rpm import NatafTransformation


###############################################################################
# Test sormBreitung
###############################################################################
def test_sormBreitung_dimLessOneCase_valueError( ):
    dim = 0

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )


def test_sormBreitung_distObjsDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    distObjs = [ X1 ]
    corrMat = np.eye( dim )
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )


def test_sormBreitung_corrMatDimMismatchCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0 ], [ 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )
    

def test_sormBreitung_corrMatNotTwoDimCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, 2.0 ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )


def test_sormBreitung_corrMatNotSymmCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.5 ], [ -0.5, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )


def test_sormBreitung_corrMatNotPositiveDefiniteCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -1.2 ], [ -1.2, 1.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )


def test_sormBreitung_corrMatDiagNotOneCase_valueError( ):
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] 
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 2.0 ] ]
    with pytest.raises( ValueError ):
        _, _, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_sormBreitung_twoNormalLinearCase_scalar( dgExists ):
    # Linear case should have the same results as FORM.
    dim = 2

    def g( X ):
        return -np.sum( X ) + 1

    dg = [ lambda X: -1, lambda X: -1 ] if dgExists else None
    X1 = stats.norm()
    X2 = stats.norm()
    distObjs = [ X1, X2 ]
    corrMat = np.eye( dim )
    calBeta, calPf, calUCoord, calXCoord = rrm.sormBreitung( dim, g, dg, distObjs, 
                                                             corrMat )
    expectedBeta = np.sqrt( 2 ) / 2
    expectedPf = stats.norm.cdf( -expectedBeta )
    expectedUCoord = [ 0.5, 0.5 ]
    expectedXCoord = [ 0.5, 0.5 ]
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
    np.testing.assert_allclose( np.round( expectedUCoord, 4 ), 
                                np.round( calUCoord, 4 ) )
    np.testing.assert_allclose( np.round( expectedXCoord, 4 ), 
                                np.round( calXCoord, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
@patch( "ffpack.rrm.firstOrderReliabilityMethod.formHLRF" )
def test_sormBreitung_twoNormalNonLinearCase_scalar( mock_formHLRF, dgExists ):
    r'''
    This example come from the reference [Choi2007] page 132.

    Here are some expected values of local variables from the refernce:
    lsfGradAtU = [ 119.8184, 124.8218 ]
    lsfHmAtU = [ [ 989.5592, 0 ],
                 [ 0, 1281.2632 ] ]
    HBH = [ [ 6.5278, -0.8423 ],
            [ -0.8423, 6.5968 ] ]

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
    mock_formHLRF.return_value = [ 2.3654, stats.norm.cdf( -2.3654 ),
                                   [ -1.6368, -1.7077 ], [ 1.8162, 1.4613 ] ]
    calBeta, calPf, _, _ = rrm.sormBreitung( dim, g, dg, distObjs, corrMat )
    expectedBeta = 2.3654
    expectedPf = 0.00222059
    np.testing.assert_allclose( np.round( expectedBeta, 4 ), np.round( calBeta, 4 ) )
    np.testing.assert_allclose( np.round( expectedPf, 4 ), np.round( calPf, 4 ) )
