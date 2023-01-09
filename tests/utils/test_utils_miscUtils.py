#!/usr/bin/env python3

from ffpack import utils
import numpy as np
import pytest



###############################################################################
# Test centralDiffWeights
###############################################################################
def test_centralDiffWeights_npLeNdivPlusOne_valueError():
    Np = 5
    ndiv = 5
    with pytest.raises( ValueError ):
        _ = utils.centralDiffWeights( Np, ndiv )

    ndiv = 6
    with pytest.raises( ValueError ):
        _ = utils.centralDiffWeights( Np, ndiv )


def test_centralDiffWeights_npIsEven_valueError():
    Np = 4
    ndiv = 1
    with pytest.raises( ValueError ):
        _ = utils.centralDiffWeights( Np, ndiv )


@pytest.mark.parametrize( "Np", [ 3, 5, 7, 9 ])
def test_centralDiffWeights_ndivIsOne_array( Np ):
    ndiv = 1
    expectedWeights = [ ]
    if Np == 3:
        expectedWeights = np.array( [ -1, 0, 1 ] ) / 2.0
    elif Np == 5:
        expectedWeights = np.array( [ 1, -8, 0, 8, -1 ] ) / 12.0
    elif Np == 7:
        expectedWeights = np.array( [ -1, 9, -45, 0, 45, -9, 1 ] ) / 60.0
    elif Np == 9:
        expectedWeights = np.array( [ 3, -32, 168, -672, 0, 
                                      672, -168, 32, -3 ] ) / 840.0
    
    calWeights = utils.centralDiffWeights( Np, ndiv )
    np.testing.assert_allclose( np.round( calWeights, 5 ), 
                                np.round( expectedWeights, 5 ) )


@pytest.mark.parametrize( "Np", [ 3, 5, 7, 9 ])
def test_centralDiffWeights_ndivIsTwo_array( Np ):
    ndiv = 2
    expectedWeights = [ ]
    if Np == 3:
        expectedWeights = np.array( [ 1, -2.0, 1 ] )
    elif Np == 5:
        expectedWeights = np.array( [ -1, 16, -30, 16, -1 ] ) / 12.0
    elif Np == 7:
        expectedWeights = np.array( [ 2, -27, 270, -490, 270, -27, 2 ] ) / 180.0
    elif Np == 9:
        expectedWeights = np.array( [ -9, 128, -1008, 8064, -14350, 
                                      8064, -1008, 128, -9 ] ) / 5040.0
    
    calWeights = utils.centralDiffWeights( Np, ndiv )
    np.testing.assert_allclose( np.round( calWeights, 5 ), 
                                np.round( expectedWeights, 5 ) )



###############################################################################
# Test derivative
###############################################################################
def test_derivative_orderLeNPlusOne_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.derivative( f, 1.0, dx=1e-6, n=3, order=3 )

    with pytest.raises( ValueError ):
        _ = utils.derivative( f, 1.0, dx=1e-6, n=4, order=3 )


def test_derivative_orderIsEven_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.derivative( f, 1.0, dx=1e-6, order=4 )


@pytest.mark.parametrize( "x0", np.linspace( -10, 10, 11 ) )
def test_derivative_linearFun_scalar( x0 ):
    f = lambda x: 2 * x 
    df = lambda x: 2

    calRst = utils.derivative( f, x0, dx=1e-6 )
    expectedRst = df( x0 )
    np.testing.assert_allclose( calRst, expectedRst )


@pytest.mark.parametrize( "x0", np.linspace( -10, 10, 11 ) )
def test_derivative_nonLinearFunCase1_scalar( x0 ):
    f = lambda x: x ** 3 + 2 * x ** 2 + 7 * x
    df = lambda x: 3 * x ** 2 + 4 * x + 7

    calRst = utils.derivative( f, x0, dx=1e-6 )
    expectedRst = df( x0 )
    np.testing.assert_allclose( calRst, expectedRst )


@pytest.mark.parametrize( "x0", np.linspace( -10, 10, 11 ) )
def test_derivative_nonLinearFunCase2_scalar( x0 ):
    f = lambda x: x ** 2 + 2 * np.exp( x ) + 7 * x
    df = lambda x: 2 * x + 2 * np.exp( x ) + 7

    calRst = utils.derivative( f, x0, dx=1e-6 )
    expectedRst = df( x0 )
    np.testing.assert_allclose( calRst, expectedRst )



###############################################################################
# Test allDerivative
###############################################################################
def test_allDerivative_nvarFloat_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.allDerivative( f, 1.0, n=1, dx=1e-6, order=1 )


def test_allDerivative_nFloat_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.allDerivative( f, 1, n=1.0, dx=1e-6, order=1 )


def test_allDerivative_orderFloat_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.allDerivative( f, 1, n=1, dx=1e-6, order=1.0 )


def test_allDerivative_orderLeNPlusOne_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.allDerivative( f, 1, n=1, dx=1e-6, order=1 )

    with pytest.raises( ValueError ):
        _ = utils.allDerivative( f, 1, n=3, dx=1e-6, order=3 )


def test_allDerivative_orderIsEven_valueError():
    f = lambda x: 2 * x 
    with pytest.raises( ValueError ):
        _ = utils.allDerivative( f, 1, n=1, dx=1e-6, order=4 )


@pytest.mark.parametrize( "X0", [ [  0.54243825, 3.58920411 ],
                                  [ -4.03059987, -1.49798558 ],
                                  [ -4.44379358, 3.52676363 ],
                                  [ -2.54625488, 4.10489644 ],
                                  [  3.4854473, 1.01969153 ],
                                  [  0.5913468, -4.33074242 ],
                                  [  3.00562433, 3.1993043 ],
                                  [  4.6750192, 1.74839702 ],
                                  [  2.05152893, -2.28757323 ],
                                  [ -3.41602264, 4.17722564 ] ] )
def test_allDerivative_firstDerivativeCase1_funcList( X0 ):
    nvar = 2
    func = lambda X: X[ 0 ] ** 3 + 2 * X[ 1 ] ** 2

    calRst = utils.allDerivative( func, nvar, n=1 )
    expectedRst = [ lambda X: 3 * X[ 0 ] ** 2,
                    lambda X: 4 * X[ 1 ] ]
    for i in range( nvar ):
        np.testing.assert_allclose( np.round( calRst[ i ]( X0 ), 4 ), 
                                    np.round( expectedRst[ i ]( X0 ), 4 ) )


@pytest.mark.parametrize( "X0", [ [  0.54243825, 3.58920411 ],
                                  [ -4.03059987, -1.49798558 ],
                                  [ -4.44379358, 3.52676363 ],
                                  [ -2.54625488, 4.10489644 ],
                                  [  3.4854473, 1.01969153 ],
                                  [  0.5913468, -4.33074242 ],
                                  [  3.00562433, 3.1993043 ],
                                  [  4.6750192, 1.74839702 ],
                                  [  2.05152893, -2.28757323 ],
                                  [ -3.41602264, 4.17722564 ] ] )
def test_allDerivative_firstDerivativeCase2_funcList( X0 ):
    nvar = 2
    func = lambda X: X[ 0 ] * X[ 1 ]

    calRst = utils.allDerivative( func, nvar, n=1 )
    expectedRst = [ lambda X: X[ 1 ],
                    lambda X: X[ 0 ] ]
    for i in range( nvar ):
        np.testing.assert_allclose( np.round( calRst[ i ]( X0 ), 4 ), 
                                    np.round( expectedRst[ i ]( X0 ), 4 ) )


@pytest.mark.parametrize( "X0", [ [  0.54243825, 3.58920411 ],
                                  [ -4.03059987, -1.49798558 ],
                                  [ -4.44379358, 3.52676363 ],
                                  [ -2.54625488, 4.10489644 ],
                                  [  3.4854473, 1.01969153 ],
                                  [  0.5913468, -4.33074242 ],
                                  [  3.00562433, 3.1993043 ],
                                  [  4.6750192, 1.74839702 ],
                                  [  2.05152893, -2.28757323 ],
                                  [ -3.41602264, 4.17722564 ] ] )
def test_allDerivative_secondDerivativeCase_funcList( X0 ):
    nvar = 2
    func = lambda X: np.sin( X[ 0 ] ) + np.cos( X[ 1 ] )

    calRst = utils.allDerivative( func, nvar, n=2 )
    expectedRst = [ lambda X: -np.sin( X[ 0 ] ),
                    lambda X: -np.cos( X[ 1 ] ) ]
    for i in range( nvar ):
        np.testing.assert_array_almost_equal( np.round( calRst[ i ]( X0 ), 4 ), 
                                              np.round( expectedRst[ i ]( X0 ), 4 ) )


@pytest.mark.parametrize( "X0", [ [  0.54243825, 3.58920411 ],
                                  [ -4.03059987, -1.49798558 ],
                                  [ -4.44379358, 3.52676363 ],
                                  [ -2.54625488, 4.10489644 ],
                                  [  3.4854473, 1.01969153 ],
                                  [  0.5913468, -4.33074242 ],
                                  [  3.00562433, 3.1993043 ],
                                  [  4.6750192, 1.74839702 ],
                                  [  2.05152893, -2.28757323 ],
                                  [ -3.41602264, 4.17722564 ] ] )
def test_allDerivative_thirdDerivativeCase_funcList( X0 ):
    nvar = 2
    func = lambda X: np.sin( X[ 0 ] ) + np.cos( X[ 1 ] )

    calRst = utils.allDerivative( func, nvar, n=3, order=5 )
    expectedRst = [ lambda X: -np.cos( X[ 0 ] ),
                    lambda X: np.sin( X[ 1 ] ) ]
    for i in range( nvar ):
        np.testing.assert_array_almost_equal( np.round( calRst[ i ]( X0 ), 4 ), 
                                              np.round( expectedRst[ i ]( X0 ), 4 ) )
