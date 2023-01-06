#!/usr/bin/env python3

from ffpack import rpm
import numpy as np
from scipy import stats
import pytest
from unittest.mock import patch


###############################################################################
# Test NatafTransformation
###############################################################################
def test_NatafTransformation_initWithDisObjsEmpty_valueError():
    rho = 0.5

    distObjs = [ ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    with pytest.raises( ValueError ):
        _ = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )


def test_NatafTransformation_initWithDisObjsDimMismatch_valueError():
    X1 = stats.norm()

    rho = 0.5

    distObjs = [ X1 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    with pytest.raises( ValueError ):
        _ = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )


def test_NatafTransformation_initWithOneDimCorrMat_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ 1.0, rho ] 

    with pytest.raises( ValueError ):
        _ = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )


def test_NatafTransformation_initWithCorrMatDiagNotOne_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 0.5, rho ], [ rho, 1.0 ] ]

    with pytest.raises( ValueError ):
        _ = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )


def test_NatafTransformation_initWithCorrMatNotSymm_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, -rho ], [ rho, 1.0 ] ]

    with pytest.raises( ValueError ):
        _ = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )


def test_NatafTransformation_initWithCorrMatNotPositiveDefinite_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = -1.2

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    with pytest.raises( ValueError ):
        _ = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )


def test_NatafTransformation_getXWithULengthMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.getX( [ 1.0 ] )


def test_NatafTransformation_getXWithUDimMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.getX( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] )


def test_NatafTransformation_getUWithXLengthMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.getU( [ 1.0 ] )


def test_NatafTransformation_getUWithXDimMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.getU( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] )


def test_NatafTransformation_pdfWithXLengthMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.pdf( [ 1.0 ] )


def test_NatafTransformation_pdfWithXDimMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.pdf( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] )


def test_NatafTransformation_cdfWithXLengthMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.cdf( [ 1.0 ] )


def test_NatafTransformation_cdfWithXDimMismatch_valueError():
    X1 = stats.norm()
    X2 = stats.norm()

    rho = 0.5

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    with pytest.raises( ValueError ):
        _ = natafDist.cdf( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] )


@pytest.mark.parametrize( "rho", [ -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, 
                                   -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, 
                                   -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, 
                                   -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 
                                   0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
                                   0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 ] )
def test_NatafTransformation_twoNormalVariablesCase_rhoZEqualCorrMat( rho ):
    X1 = stats.norm()
    X2 = stats.norm()

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    np.testing.assert_allclose( corrMat, natafDist.rhoZ )


@pytest.mark.parametrize( "rho", [ 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 
                                   0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 
                                   0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 ] )
def test_NatafTransformation_threeNormalVariablesCase_rhoZEqualCorrMat( rho ):
    X1 = stats.norm()
    X2 = stats.norm()
    X3 = stats.norm()

    distObjs = [ X1, X2, X3 ]
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho ], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    np.testing.assert_allclose( corrMat, natafDist.rhoZ )


def test_NatafTransformation_normAndExpVariablesCase_2dRhoZ( ):
    X1 = stats.norm()
    X2 = stats.expon()
    distObjs = [ X1, X2 ]

    rho = -0.8
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.88574225 ], [ -0.88574225, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.55358891 ], [ -0.55358891, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.22143556], [ -0.22143556, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.22143556], [ 0.22143556, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.55358891 ], [ 0.55358891, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.8
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.88574225 ], [ 0.88574225, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )


def test_NatafTransformation_normAndGammaVariablesCase_2dRhoZ( ):
    X1 = stats.norm()
    X2 = stats.gamma( 2 )
    distObjs = [ X1, X2 ]

    rho = -0.8
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.84397443 ], [ -0.84397443, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.52748402 ], [ -0.52748402, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.21099361 ], [ -0.21099361, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.21099361 ], [ 0.21099361, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.52748402 ], [ 0.52748402, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.8
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.84397443 ], [ 0.84397443, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )


def test_NatafTransformation_expAndGammaVariablesCase_2dRhoZ( ):
    X1 = stats.expon()
    X2 = stats.gamma( 2 )
    distObjs = [ X1, X2 ]

    rho = -0.7
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.95687299 ], [ -0.95687299, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.64842439], [ -0.64842439, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.24271826 ], [ -0.24271826, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.22564528 ], [ 0.22564528, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.53835575 ], [ 0.53835575, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.7
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.7326582 ], [ 0.7326582, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )


def test_NatafTransformation_normAndGumbelVariablesCase_2dRhoZ( ):
    X1 = stats.norm()
    X2 = stats.gumbel_r( 2 )
    distObjs = [ X1, X2 ]

    rho = -0.7
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.72204823 ], [ -0.72204823, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.51574873 ], [ -0.51574873, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = -0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.20629949 ], [ -0.20629949, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.2
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.20629949 ], [ 0.20629949, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.5
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.51574873 ], [ 0.51574873, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.7
    corrMat = [ [ 1.0, rho ], [ rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.72204823 ], [ 0.72204823, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )


def test_NatafTransformation_normExpAndGumbelVariablesCase_2dRhoZ( ):
    X1 = stats.norm()
    X2 = stats.gumbel_r( 2 )
    X3 = stats.expon()
    distObjs = [ X1, X2, X3 ]

    rho = -0.2
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.20629949, -0.22143556 ], 
                     [ -0.20629949, 1.0, -0.23466674 ],
                     [ -0.22143556, -0.23466674, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.2
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.20629949, 0.22143556 ], 
                     [ 0.20629949, 1.0, 0.22265972 ],
                     [ 0.22143556, 0.22265972, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.5
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.51574873, 0.55358891 ], 
                     [ 0.51574873, 1.0, 0.53709868 ],
                     [ 0.55358891, 0.53709868, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.7
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.72204823, 0.77502447 ], 
                     [ 0.72204823, 1.0, 0.73530146 ],
                     [ 0.77502447, 0.73530146, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )


def test_NatafTransformation_expGammaAndGumbelVariablesCase_2dRhoZ( ):
    X1 = stats.gumbel_r( 2 )
    X2 = stats.expon()
    X3 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3 ]

    rho = -0.2
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, -0.23466674, -0.2216493 ], 
                     [ -0.23466674, 1.0, -0.24271826 ],
                     [ -0.2216493, -0.24271826, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.2
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.22265972, 0.21385972 ], 
                     [ 0.22265972, 1.0, 0.22564528 ],
                     [ 0.21385972, 0.22564528, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.5
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.53709868, 0.52143531 ], 
                     [ 0.53709868, 1.0, 0.53835575 ],
                     [ 0.52143531, 0.53835575, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )

    rho = 0.7
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    expectedRhoZ = [ [ 1.0, 0.73530146, 0.71847694 ], 
                     [ 0.73530146, 1.0, 0.7326582 ],
                     [ 0.71847694, 0.7326582, 1.0 ] ]
    np.testing.assert_allclose( np.round( expectedRhoZ, 5 ), 
                                np.round( natafDist.rhoZ, 5 ) )


def test_NatafTransformation_normalExpAndGammaVariablesPositiveRho_outputScalar( ):
    X1 = stats.norm()
    X2 = stats.expon()
    X3 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3 ]

    rho = 0.5
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    X = [ 2.64875507, 1.31376497, 3.17128283 ]
    calU, calJX2U = natafDist.getU( X )
    calX, calJU2X = natafDist.getX( calU )
    calPdf = natafDist.pdf( X )
    calCdf = natafDist.cdf( X )
    expectedU = [ 2.64875507, -1.02053763, -0.20170925 ]
    expectedJX2U = [ [ 1.0, 0.0, 0.0 ], 
                     [ 0.67943202, 1.02210185, 0.0 ], 
                     [ 1.02202484, 0.57314287, 1.54306806 ] ]
    expectedJU2X = [ [ 1.0, 0.0, 0.0 ], 
                     [ -0.66474004, 0.97837608, 0.0 ], 
                     [ -0.41542809, -0.36339893, 0.64805955 ] ]
    expectedPdf = 0.0004366062745374653
    expectedCdf = 0.6104865261872772
    
    np.testing.assert_allclose( np.round( expectedU, 5 ), np.round( calU, 5 ) )
    np.testing.assert_allclose( np.round( X, 5 ), np.round( calX, 5 ) )
    np.testing.assert_allclose( np.round( expectedJX2U, 5 ), np.round( calJX2U, 5 ) )
    np.testing.assert_allclose( np.round( expectedJU2X, 5 ), np.round( calJU2X, 5 ) )
    np.testing.assert_allclose( np.round( expectedPdf, 6 ), np.round( calPdf, 6 ) )
    np.testing.assert_allclose( np.round( expectedCdf, 4 ), np.round( calCdf, 4 ) )


def test_NatafTransformation_normalExpAndGammaVariablesNegativeRho_outputScalar( ):
    X1 = stats.norm()
    X2 = stats.expon()
    X3 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3 ]

    rho = -0.2
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    X = [ 0.67601087, 1.77035951, 0.54476353 ]
    calU, calJX2U = natafDist.getU( X )
    calX, calJU2X = natafDist.getX( calU )
    calPdf = natafDist.pdf( X )
    calCdf = natafDist.cdf( X )
    expectedU = [ 0.67601087, 1.13085826, -0.83795346 ]
    expectedJX2U = [ [ 1.0, 0.0, 0.0 ], 
                     [ -0.3294289, 1.4507644, 0.0 ], 
                     [ -0.12065027, -0.16972062, 0.53255612 ] ]
    expectedJU2X = [ [ 1.0, 0.0, 0.0 ], 
                     [ 0.22707264, 0.68929179, 0.0 ], 
                     [ 0.29891531, 0.2196708, 1.87773638 ] ]
    expectedPdf = 0.022124746073085712
    expectedCdf = 0.08251737968730118
    
    np.testing.assert_allclose( np.round( expectedU, 5 ), np.round( calU, 5 ) )
    np.testing.assert_allclose( np.round( X, 5 ), np.round( calX, 5 ) )
    np.testing.assert_allclose( np.round( expectedJX2U, 5 ), np.round( calJX2U, 5 ) )
    np.testing.assert_allclose( np.round( expectedJU2X, 5 ), np.round( calJU2X, 5 ) )
    np.testing.assert_allclose( np.round( expectedPdf, 6 ), np.round( calPdf, 6 ) )
    np.testing.assert_allclose( np.round( expectedCdf, 4 ), np.round( calCdf, 4 ) )


def test_NatafTransformation_normalExpAndGammaVariablesZeroRho_outputScalar( ):
    X1 = stats.norm()
    X2 = stats.expon()
    X3 = stats.gamma( 2 )
    distObjs = [ X1, X2, X3 ]

    rho = 0.0
    corrMat = [ [ 1.0, rho, rho ], [ rho, 1.0, rho], [ rho, rho, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    X = [ 0.55587583, 0.95075328, 0.22449628 ]
    calU, calJX2U = natafDist.getU( X )
    calX, calJU2X = natafDist.getX( calU )
    calPdf = natafDist.pdf( X )
    calCdf = natafDist.cdf( X )
    expectedU = [ 0.55587583, 0.28858416, -2.01931569 ]
    expectedJX2U = [ [ 1.0, 0.0, 0.0 ], 
                     [ 0.0, 0.99022245, 0.0 ], 
                     [ 0.0, 0.0, 0.28956831 ] ]
    expectedJU2X = [ [ 1.0, 0.0, 0.0 ], 
                     [ 0.0, 1.00987409, 0.0 ], 
                     [ 0.0, 0.0, 3.45341652 ] ]
    expectedPdf = 0.04643286226589014
    expectedCdf = 0.03343666723873232
    
    np.testing.assert_allclose( np.round( expectedU, 5 ), np.round( calU, 5 ) )
    np.testing.assert_allclose( np.round( X, 5 ), np.round( calX, 5 ) )
    np.testing.assert_allclose( np.round( expectedJX2U, 5 ), np.round( calJX2U, 5 ) )
    np.testing.assert_allclose( np.round( expectedJU2X, 5 ), np.round( calJU2X, 5 ) )
    np.testing.assert_allclose( np.round( expectedPdf, 6 ), np.round( calPdf, 6 ) )
    np.testing.assert_allclose( np.round( expectedCdf, 4 ), np.round( calCdf, 4 ) )


@patch( "numpy.random.randn" )
def test_NatafTransformation_sampleDataPoint_1dArray( mock_randn ):
    mock_randn.return_value = [ 1.0, 1.0 ]
    X1 = stats.norm()
    X2 = stats.norm()

    distObjs = [ X1, X2 ]
    corrMat = [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ]

    natafDist = rpm.NatafTransformation( distObjs=distObjs, corrMat=corrMat )
    calSample = natafDist.getSample()
    expectedSample = [ 1.0, 1.0 ]
    np.testing.assert_allclose( expectedSample, calSample )


    mock_randn.return_value = [ 0.5, 0.5 ]
    calSample = natafDist.getSample()
    expectedSample = [ 0.5, 0.5 ]
    np.testing.assert_allclose( expectedSample, calSample )
