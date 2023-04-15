#!/usr/bin/env python3

'''
This module implements the Nataf distribution. The Gaussian quadrature and root
finding algorithm are used to determine the integral resutls and rho' of Eq.(12)
in reference [1]_. Reference [2]_ uses the Gauss–Legendre quadrature to calculate
the integral results. With the transformation, Gauss–Hermite quadrature can also be
used to calculate the integral results, as indicated in reference [3]_. However, 
the iteration method proposed in reference [3]_ requires Cholesky decomposition 
in the iteration. This module follows the Nataf transformation method implemented 
in reference [2]_.


.. [1] Liu, P.L. and Der Kiureghian, A., 1986. Multivariate distribution models 
       with prescribed marginals and covariances. Probabilistic engineering 
       mechanics, 1(2), pp.105-112.

.. [2] Ehre, M., Geyer, S., Kamariotis, A., Papaioannou, I., Sardi, L., 2022. 
       Documentation of the ERA Distribution Classes.

.. [3] Li, H., Lü, Z. and Yuan, X., 2008. Nataf transformation based point 
       estimate method. Chinese Science Bulletin, 53(17), pp.2586-2592.

'''

import numpy as np
from scipy import stats, optimize


class NatafTransformation:
    '''
    Nataf distribution for correlated marginal distributions.
    '''
    def __init__( self, distObjs, corrMat, quadDeg=99, quadRange=8 ):
        '''
        Initialize the Nataf distribution.
        
        Parameters
        ----------
        distObjs: array_like of distributions
            Marginal distribution objects. It should be the freezed distribution 
            objects with pdf, cdf, ppf. We recommend to use scipy.stats functions.
        corrMat: 2d matrix
            Correlation matrix of the marginal distributions.
        quadDeg: integer
            Quadrature degree.
        quadRange: scalar
            Quadrature range. The integral will be performed in the range
            [ -quadRange, quadRange ].
        
        Raises
        ------
        ValueError
            If distObjs is empty.
            If dimensions are not match for distObjs and corrMat.
            If corrMat is not 2d matrix.
            If corrMat is not positive definite.
            If corrMat is not symmetric.
            If corrMat diagonal is not 1.
        
        Examples
        --------
        >>> from ffpack.rpm import NatafTransformation
        >>> distObjs = [ stats.norm(), stats.norm() ]
        >>> corrMat = [ [ 1.0, 0.5 ], [ 0.5, 1.0 ] ]
        >>> natafDist = NatafTransformation( distObjs=distObjs, corrMat=corrMat )
        '''

        if len( distObjs ) == 0:
            raise ValueError( "distObjs cannot be empty" )
        
        corrMat = np.array( corrMat )
        if not np.all( np.diag( corrMat ) == 1 ):
            raise ValueError( "diagonals of corrMat should be 1" )

        if len( distObjs ) != corrMat.shape[ 0 ] or \
           len( distObjs ) != corrMat.shape[ 1 ]:
            raise ValueError( "dimensions are mismatched for distObjs and corrMat" )
        
        if corrMat.ndim != 2:
            raise ValueError( "corrMat should be 2d matrix" )

        if not np.array_equal( corrMat, corrMat.T ):
            raise ValueError( "corrMat should be symmetric" )
        
        try:
            _ = np.linalg.cholesky( corrMat )
        except np.linalg.LinAlgError:
            raise ValueError( "corrMat should be positive definite" )

        self.distObjs = distObjs
        self.rhoX = np.array( corrMat )
        self.dim = len( distObjs )
        self.deg = quadDeg
        self.ran = quadRange

        self.rhoZ = np.identity( self.dim )

        intPoints, intWeights = np.polynomial.legendre.leggauss( deg=self.deg )
        intPoints = self.ran * intPoints
        intWeights = self.ran * intWeights
        intPointsXX, intPointsYY = np.meshgrid( intPoints, intPoints )

        int2DWeights = np.tile( intWeights, [ len( intWeights ), 1 ] ) * \
            np.tile( intWeights, [ len( intWeights ), 1 ] ).T
        
        def stdBivariateNormalPdf( x, y, rho ):
            if rho >= 1.0:
                rho = 1 - np.finfo(float).eps
            if rho <= -1.0:
                rho = -1 + np.finfo(float).eps
            return 1 / ( 2 * np.pi * np.sqrt( 1 - rho**2) ) * \
                np.exp( -1 / ( 2 * ( 1 - rho**2 ) ) * 
                        ( x**2 - 2 * rho * x * y + y**2 ) )
        
        def solve( func, x0 ):
            rst = optimize.fsolve( func=func, 
                                   x0=x0,
                                   full_output=True )
            return rst
        
        for i in range( self.dim ):
            for j in range( i + 1, self.dim ):
                if self.rhoX[ i, j ] == 0:
                    continue

                termI = ( self.distObjs[ i ].ppf( stats.norm.cdf( intPointsXX ) ) - 
                          self.distObjs[ i ].mean() ) / self.distObjs[ i ].std()
                termI[ termI == np.inf ] = np.sqrt( np.finfo(float).max ) - 1
                termI[ termI == -np.inf ] = -np.sqrt( np.finfo(float).max ) + 1

                termJ = ( self.distObjs[ j ].ppf( stats.norm.cdf( intPointsYY ) ) - 
                          self.distObjs[ j ].mean() ) / self.distObjs[ j ].std()
                termJ[ termJ == np.inf ] = np.sqrt( np.finfo(float).max ) - 1
                termJ[ termJ == -np.inf ] = -np.sqrt( np.finfo(float).max ) + 1

                intg = termI * termJ * int2DWeights

                def func( rho0 ):
                    lhs = self.rhoX[ i, j ]
                    rhs = ( intg * (stdBivariateNormalPdf( intPointsXX, 
                                                           intPointsYY, 
                                                           rho=rho0 ) ) ).sum()
                    return lhs - rhs
                
                rst = solve( func=func, x0=self.rhoX[ i, j ] )
                if rst[ 2 ] == 1:
                    self.rhoZ[ i, j ] = rst[ 0 ]
                    self.rhoZ[ j, i ] = self.rhoZ[ i, j ]
                    continue

                rst = solve( func=func, x0=-self.rhoX[ i, j ] )
                if rst[ 2 ] == 1:
                    self.rhoZ[ i, j ] = rst[ 0 ]
                    self.rhoZ[ j, i ] = self.rhoZ[ i, j ]
                    continue

                # If algorithm cannot determine the root with the previous two 
                # starting points, try each starting point in ( -1.0, 1.0 ).
                for k in np.linspace( -0.9, 0.9, 19 ):
                    rst = solve( func=func, x0=k )
                    if rst[ 2 ] == 1:
                        self.rhoZ[ i, j ] = rst[ 0 ]
                        self.rhoZ[ j, i ] = self.rhoZ[ i, j ]
                        break
                
                raise ValueError( "Nataf transformation cannot be performed" )

        try:
            self.L = np.linalg.cholesky( self.rhoZ )
        except np.linalg.LinAlgError:
            raise ValueError( "Nataf transformation cannot be performed" )

    def getU( self, X ):
        '''
        Get data point in U space and Jacobian.
        
        Parameters
        ----------
        X: 1d array
            Data point in X space.
        
        Returns
        -------
        U: 1d array
            Data point in U space.
        J: 2d matrix
            Jacobian from X space to U space.

        Raises
        ------
        ValueError
            If length of X does not match dim.
            If X is not 1d array.
        
        Notes
        -----
        X -> Z -> U

        Examples
        --------
        >>> X = [ 0.5, 1.5 ]
        >>> U, J = natafDist.getU( X )
        '''
        if len( X ) != self.dim:
            raise ValueError( "length of X should be the same as dim" )
        
        X = np.array( X )
        if X.ndim != 1:
            raise ValueError( "X should be 1d array")

        # X -> Z
        Z = np.zeros_like( X )
        for d in range( self.dim ):
            Z[ d ] = stats.norm.ppf( self.distObjs[ d ].cdf( X[ d ] ) )

        # Z -> U
        U = np.linalg.solve( self.L, Z.T ).T
        
        # Jacobian = diag[ phi( y_i ) / f( x_i )] * L
        diagMat = np.zeros( ( self.dim, self.dim ) )
        for d in range( self.dim ):
            diagMat[ d, d ] = stats.norm.pdf( Z[ d ] ) / \
                self.distObjs[ d ].pdf( X[ d ] )
        J = np.dot( diagMat, self.L )
        
        return U, J

    def getX( self, U ):
        '''
        Get data point in X space and Jacobian.
        
        Parameters
        ----------
        U: 1d array
            Data point in U space.
        
        Returns
        -------
        X: 1d array
            Data point in X space.
        J: 2d matrix
            Jacobian from U space to X space.

        Raises
        ------
        ValueError
            If length of U does not match dim.
            If U is not 1d array.
        
        Notes
        -----
        U -> Z -> X

        Examples
        --------
        >>> U = [ 0.5, 1.5 ]
        >>> X, J = natafDist.getX( U )
        '''
        if len( U ) != self.dim:
            raise ValueError( "length of U should be the same as dim" )
        
        U = np.array( U )
        if U.ndim != 1:
            raise ValueError( "U should be 1d array")

        # U -> Z
        Z = np.dot( self.L, U.T ).T

        # Z -> X
        X = np.zeros_like( U )
        for d in range( self.dim ):
            X[ d ] = self.distObjs[ d ].ppf( stats.norm.cdf( Z[ d ] ) )

        # Jacobina = L ^ ( -1 ) * diag[ f( x_i ) / phi( y_i ) ] 
        diagMat = np.zeros( ( self.dim, self.dim ) )
        for d in range( self.dim ):
            diagMat[ d, d ] = self.distObjs[ d ].pdf( X[ d ] ) / \
                stats.norm.pdf( Z[ d ] )
        J = np.linalg.solve( self.L, diagMat )

        return X, J 
    
    def pdf( self, X ):
        '''
        Get pdf value for X.
        
        Parameters
        ----------
        X: 1d array
            Data point in X space.
        
        Returns
        -------
        rst: scalar
            Value of probability density function at data point X.

        Raises
        ------
        ValueError
            If length of X does not match dim.
            If X is not 1d array.
        
        Examples
        --------
        >>> X = [ 0.5, 1.5 ]
        >>> rst = natafDist.pdf( X )
        '''
        if len( X ) != self.dim:
            raise ValueError( "length of X should be the same as dim" )

        X = np.array( X )
        if X.ndim != 1:
            raise ValueError( "X should be 1d array")

        # X -> Z
        Z = np.zeros_like( X )
        phi = np.zeros_like( X ) 
        y = np.zeros_like( X )
        mu = np.zeros( self.dim )
        std = np.array( [ dist.std() for dist in self.distObjs ] )
        cov = np.diag( std ) @ self.rhoZ @ np.diag( std )
        mv = stats.multivariate_normal( mean=mu, cov=cov )
        for d in range( self.dim ):
            Z[ d ] = stats.norm.ppf( self.distObjs[ d ].cdf( X[ d ] ) )
            phi[ d ] = stats.norm.pdf( Z[ d ] )
            y[ d ] = self.distObjs[ d ].pdf( X[ d ] )
        rst = 0
        if not np.isclose( np.prod( phi ), 0 ):
            rst = np.prod( y ) / np.prod( phi ) * mv.pdf( Z )
        return rst
    
    def cdf( self, X ):
        '''
        Get cdf value for X.
        
        Parameters
        ----------
        X: 1d array
            Data point in X space.
        
        Returns
        -------
        rst: scalar
            Value of cumulative distribution function at data point X.

        Raises
        ------
        ValueError
            If length of X does not match dim.
            If X is not 1d array.
        
        Examples
        --------
        >>> X = [ 0.5, 1.5 ]
        >>> rst = natafDist.cdf( X )
        '''
        if len( X ) != self.dim:
            raise ValueError( "length of X should be the same as dim" )

        X = np.array( X )
        if X.ndim != 1:
            raise ValueError( "X should be 1d array")

        # X -> Z
        Z = np.zeros_like( X )
        mu = np.zeros( self.dim )
        std = np.array( [ dist.std() for dist in self.distObjs ] )
        cov = np.diag( std ) @ self.rhoZ @ np.diag( std )
        mv = stats.multivariate_normal( mean=mu, cov=cov )
        for d in range( self.dim ):
            Z[ d ] = stats.norm.ppf( self.distObjs[ d ].cdf( X[ d ] ) )

        rst = mv.cdf( Z )
        return rst

    def getSample( self ):
        '''
        Get a sample in X space from Nataf distribution
        
        Returns
        -------
        X: 1d array
            Data point in X space.

        Examples
        --------
        >>> X = natafDist.getSample()
        '''
        U = np.random.randn( self.dim )
        X, _ = self.getX( U )
        return X
