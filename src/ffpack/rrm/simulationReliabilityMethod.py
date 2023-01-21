#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
from ffpack.rpm import metropolisHastings, nataf

def subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                      maxSubsets, probLevel=0.1, quadDeg=99, quadRange=8 ):
    '''
    Second order reliability method based on Breitung algorithm.

    Parameters
    ----------
    dim: integer
        Space dimension ( number of random variables ).
    g: function
        Limit state function. It will be called like g( [ x1, x2, ... ] ).
    distObjs: array_like of distributions
        Marginal distribution objects. It should be the freezed distribution 
        objects with pdf, cdf, ppf. We recommend to use scipy.stats functions.
    corrMat: 2d matrix
        Correlation matrix of the marginal distributions.
    numSamples: integer
        Number of samples in each subset.
    maxSubsets: scalar
        Maximum number of subsets used to compute the failure probability.
    probLevel: scalar, optional
        Probability level for intermediate subsets.
    quadDeg: integer, optional
        Quadrature degree for Nataf transformation
    quadRange: scalar, optional
        Quadrature range for Nataf transformation. The integral will be performed 
        in the range [ -quadRange, quadRange ].
    
    Returns
    -------
    pf: scalar
        Probability of failure.
    
    Raises
    ------
    ValueError
        If the dim is less than 1.
        If the dim does not match the disObjs and corrMat.
        If corrMat is not 2d matrix.
        If corrMat is not positive definite.
        If corrMat is not symmetric.
        If corrMat diagonal is not 1.

    Notes
    -----
    If dg is None, the numerical differentiation will be used. The tolerance of the 
    numerical differentiation can be changed in globalConfig.

    Examples
    --------
    >>> from ffpack.rrm import subsetSimulation
    >>> dim = 2
    >>> g = lambda X: -np.sum( X ) + 1
    >>> distObjs = [ stats.norm(), stats.norm() ]
    >>> corrMat = np.eye( dim )
    >>> numSamples, maxSubsets = 500, 10
    >>> pf = subsetSimulation( dim, g, distObjs, corrMat, numSamples, maxSubsets )
    '''
    # Check edge cases

    # Perform Nataf transformation
    natafTrans = nataf.NatafTransformation( distObjs=distObjs, corrMat=corrMat,
                                            quadDeg=quadDeg, quadRange=quadRange )

    # Define paramters for MCMC sampling
    def tpdf( x ):
        return norm.pdf( x )
    
    targetPdf = [ tpdf ] * dim
    
    def pcs( x ):
        return x - 0.5 + np.random.uniform()
    
    proposalCSampler = [ pcs ] * dim 

    def sampleDomainFunc( curU, nxtU, **kwargs ):
        lsfFunc = kwargs[ 'lsfFunc' ]
        lsfLevel = kwargs[ 'lsfLevel' ]
        natafTrans = kwargs[ 'natafTrans' ]           
        if not np.allclose( curU, nxtU ):
            nxtX, _ = natafTrans.getX( nxtU )
            return lsfFunc( nxtX ) < lsfLevel
        else:
            return False

    # Create each chain for each sample
    numChains = numSamplesRemained = int( probLevel * numSamples )
    numSamplesEachChain = int( 1.0 / probLevel )

    # Allocate space
    curSamples = np.zeros( [ numSamples, dim ] )
    curLsfValues = np.zeros( numSamples )
    allProbs = np.zeros( maxSubsets )
    
    # Use curde Monte Carlo to run the first iteration 
    curSamples[ : ] = np.random.normal( loc=0, scale=1.0, size=( numSamples, dim  ) )
    for idx, U in enumerate( curSamples ):
        X, _ = natafTrans.getX( U )
        # X = U
        curLsfValues[ idx ] = g( X )

    numSteps = 0
    while numSteps < maxSubsets:
        print(" each loop ")

        lsfIdx = np.argsort( curLsfValues )
        curSamples[ : ] = curSamples[ lsfIdx[ : ] ]
        curLsfValues[ : ] = curLsfValues[ lsfIdx[ : ] ]

        # intermediate level
        lsfLevel = curLsfValues[ numSamplesRemained - 1 ]

        print( lsfLevel )
        if lsfLevel <= 0:
            lsfLevel = 0
            allProbs[ numSteps ] = ( curLsfValues <= 0 ).sum() / numSamples
        else:
            allProbs[ numSteps ] = probLevel

        curIdx = numSamplesRemained
        # Sample each chain
        for i in range( numChains ):
            curU = curSamples[ i ].tolist()

            auMMHSampler = metropolisHastings.\
                AuModifiedMHSampler( initialVal=curU, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler,
                                     sampleDomain=sampleDomainFunc,
                                     lsfFunc=g,
                                     lsfLevel=lsfLevel,
                                     natafTrans=natafTrans )

            for _ in range( numSamplesEachChain - 1 ):
                curU = auMMHSampler.getSample()
                # nxtU = np.zeros( dim )
                # for d in range( dim ):
                #     nxtU[ d ] = proposalCSampler[ d ]( curU[ d ] )
                #     fcur = targetPdf[ d ]( curU[ d ] )
                #     fcandi = targetPdf[ d ]( nxtU[ d ] )
                #     u = np.random.uniform()
                #     if u > min( 1, fcandi / fcur ):
                #         nxtU[ d ] = curU[ d ]

                # if not np.allclose( nxtU, curU ):
                #     nxtX, _ = natafTrans.getX( nxtU )
                #     if g( nxtX ) < lsfLevel:
                #         curU = nxtU

                curX, _ = natafTrans.getX( curU )
                curSamples[ curIdx ] = curU
                curLsfValues[ curIdx ] = g( curX )
                curIdx += 1

        numSteps += 1
        if lsfLevel <= 0:
            break

    pf = np.prod( allProbs[ : numSteps ] )
    print( allProbs )
    print( f"{pf:.8f}" )
