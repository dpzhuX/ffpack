#!/usr/bin/env python3

from ffpack import rrm
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test FOSM
###############################################################################
def test_FOSM_dimLessThanOneCase_valueError():
    dim = 0

    def g( X ): 
        return 3 * X[ 0 ] - 2 * X[ 1 ]

    dg = [ lambda X: 3, lambda X: -2 ]
    mus = [ 1, 1 ]
    sigmas = [ 3, 4 ]

    with pytest.raises( ValueError ):
        _, _ = rrm.FOSM( dim, g, dg, mus, sigmas )


def test_FOSM_musDimIncorrectCase_valueError():
    dim = 2

    def g( X ): 
        return 3 * X[ 0 ] - 2 * X[ 1 ]

    dg = [ lambda X: 3, lambda X: -2 ]
    mus = [ 1 ]
    sigmas = [ 3, 4 ]

    with pytest.raises( ValueError ):
        _, _ = rrm.FOSM( dim, g, dg, mus, sigmas )


def test_FOSM_sigmasDimIncorrectCase_valueError():
    dim = 2

    def g( X ): 
        return 3 * X[ 0 ] - 2 * X[ 1 ]

    dg = [ lambda X: 3, lambda X: -2 ]
    mus = [ 1, 1 ]
    sigmas = [ 3 ]

    with pytest.raises( ValueError ):
        _, _ = rrm.FOSM( dim, g, dg, mus, sigmas )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_FOSM_twoVariablesCase_scalar( dgExists ):
    dim = 2

    def g( X ): 
        return 3 * X[ 0 ] - 2 * X[ 1 ]

    dg = [ lambda X: 3, lambda X: -2 ] if dgExists else None
    mus = [ 1, 1 ]
    sigmas = [ 3, 4 ]

    beta, pf = rrm.FOSM( dim, g, dg, mus, sigmas )
    np.testing.assert_allclose( beta, 1 / np.sqrt( 81 + 64 ) )
    np.testing.assert_allclose( np.round( pf, 4 ), np.round( 0.46692577, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_FOSM_threeVariablesCase_scalar( dgExists ):
    dim = 3

    def g( X ): 
        return -54 * X[ 0 ] - 5832 * X[ 1 ] + 80 * X[ 2 ]

    dg = [ lambda X: -54, lambda X: -5832, lambda X: 80 ] if dgExists else None
    mus = [ 10.2, 0.25, 40.3 ]
    sigmas = [ 1.12, 0.025, 4.64 ]

    beta, pf = rrm.FOSM( dim, g, dg, mus, sigmas )
    np.testing.assert_allclose( np.round( beta, 2 ), 3.01 )
    np.testing.assert_allclose( np.round( pf, 4 ), np.round( 0.00130624, 4 ) )


@pytest.mark.parametrize( "dgExists", [ 0, 1 ] )
def test_FOSM_fourVariablesCase_scalar( dgExists ):
    dim = 4

    def g( X ): 
        return 19 * X[ 0 ] * X[ 1 ] - 0.59 * ( X[ 0 ] * X[ 1 ] ) ** 2 / \
            ( 12 * X[ 2 ] ) - X[ 3 ]

    dg = [ 
        lambda X: 19 * X[ 1 ] - 0.59 * ( 2 * X[ 0 ] * X[ 1 ] ** 2) / ( 12 * X[ 2 ]), 
        lambda X: 19 * X[ 0 ] - 0.59 * ( 2 * X[ 0 ] ** 2 * X[ 1 ]) / ( 12 * X[ 2 ]), 
        lambda X: 0.59 * ( X[ 0 ] * X[ 1 ] ) ** 2 / ( 12 * X[ 2 ] ** 2 ),
        lambda X: -1 ] if dgExists else None
    mus = [ 4.08, 44, 3.12, 2052 ]
    sigmas = [ 0.08, 4.62, 0.44, 246 ] 

    beta, pf = rrm.FOSM( dim, g, dg, mus, sigmas )
    np.testing.assert_allclose( np.round( beta, 2 ), 2.35 )
    np.testing.assert_allclose( np.round( pf, 4 ), np.round( 0.00938671, 4 ) )
