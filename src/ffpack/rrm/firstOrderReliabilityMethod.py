#!/usr/bin/env python3

import numpy as np
import scipy as sp
from ffpack.config import globalConfig 
from ffpack.rpm import nataf

def form( dim, g, dg, distObjs, corrMat, iter=1000, tol=1e-6 ):

    def partialDerivative( func, var=0, points=[ ] ):
        args = points[ : ]

        def wraps( x ):
            args[ var ] = x
            return func( args )
        return sp.misc.derivative( wraps, points[ var ], 
                                   dx=1 / np.power( 10, globalConfig.dtol ) )
    
    def dgWrap( g, var=0 ):
        def dgi( mus ):
            return partialDerivative( g, var=var, points=mus )
        return dgi

    if dg is None:
        dg = [ dgWrap( g, i ) for i in range( dim ) ]

    natafTrans = nataf.NatafTransformation( distObjs=distObjs, corrMat=corrMat )

    Us = np.ones( [ iter + 1, dim ] )
    alphas = np.zeros_like( Us )
    betas = np.zeros( iter + 1 )
    idx = 1
    for idx in range( 1, iter + 1):
        # J: U -> X is partialX / partialU
        X, J = natafTrans.getX( Us[ idx - 1 ] )

        a = np.array( [ dgi( X ) for dgi in dg ] )
        gPrime = np.linalg.solve( J.T, a )
        gPrime = gPrime.T.flatten()
        gPrimeNorm = np.linalg.norm( gPrime )

        betas[ idx ] = ( g( X ) - np.dot( Us[ idx - 1 ], gPrime ) ) / gPrimeNorm

        alphas[ idx ] = gPrime / gPrimeNorm

        Us[ idx ] = -betas[ idx ] * alphas[ idx ]
        if np.linalg.norm( Us[ idx ] - Us[ idx - 1 ] ) < tol:
            break
    
    rstBeta = betas[ idx ]
    rstPf = sp.stats.norm.cdf( -rstBeta )
    rstUCoord = Us[ idx ]
    rstXCoord, _ = natafTrans.getX( rstUCoord )
    return rstBeta, rstPf, rstUCoord, rstXCoord
