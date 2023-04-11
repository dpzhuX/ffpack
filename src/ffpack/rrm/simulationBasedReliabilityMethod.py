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
    allLsfValue: array_like
        Values of limit state function in each subset.
    allUSamples: array_like
        Samples of U space in each subset.
    allXSamples: array_like
        Samples of X space in each subset.
    
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
    Nataf transformation is used for the marginal distributions.

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
    if dim < 1:
        raise ValueError( "dim cannot be less than 1" )

    corrMat = np.array( corrMat, dtype=float )
    if not np.all( np.diag( corrMat ) == 1 ):
        raise ValueError( "diagonals of corrMat should be 1" )

    if len( distObjs ) != dim or corrMat.shape[ 0 ] != dim \
            or corrMat.shape[ 1 ] != dim: 
        raise ValueError( "length of distObjs and corrMat should be dim" )

    if corrMat.ndim != 2:
        raise ValueError( "corrMat should be 2d matrix" )

    if not np.array_equal( corrMat, corrMat.T ):
        raise ValueError( "corrMat should be symmetric" )
    
    try:
        _ = np.linalg.cholesky( corrMat )
    except np.linalg.LinAlgError:
        raise ValueError( "corrMat should be positive definite" )

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
    curUSamples = np.zeros( [ numSamples, dim ] )
    curXSamples = np.zeros( [ numSamples, dim ] )
    curLsfValues = np.zeros( numSamples )
    allProbs = np.zeros( maxSubsets )
    allUSamples = np.zeros( [ maxSubsets, numSamples, dim ])
    allXSamples = np.zeros( [ maxSubsets, numSamples, dim ])
    allLsfValues = np.zeros( [ maxSubsets, numSamples ] )
    
    # Use curde Monte Carlo to run the first iteration 
    curUSamples[ : ] = np.random.normal( loc=0, scale=1.0, size=( numSamples, dim  ) )
    for idx, curU in enumerate( curUSamples ):
        curX, _ = natafTrans.getX( curU )
        curXSamples[ idx ] = curX
        curLsfValues[ idx ] = g( curX )

    numSteps = 0
    while numSteps < maxSubsets:
        lsfIdx = np.argsort( curLsfValues )
        curUSamples[ : ] = curUSamples[ lsfIdx[ : ] ]
        curXSamples[ : ] = curXSamples[ lsfIdx[ : ] ]
        curLsfValues[ : ] = curLsfValues[ lsfIdx[ : ] ]
        
        # Save sample and values in each subset
        allUSamples[ numSteps, :, : ] = curUSamples
        allXSamples[ numSteps, :, : ] = curXSamples
        allLsfValues[ numSteps, : ] = curLsfValues

        # Intermediate level
        lsfLevel = curLsfValues[ numSamplesRemained - 1 ]
        if lsfLevel <= 0:
            lsfLevel = 0
            allProbs[ numSteps ] = ( curLsfValues <= 0 ).sum() / numSamples
        else:
            allProbs[ numSteps ] = probLevel

        curIdx = numSamplesRemained
        # Sample each chain
        for i in range( numChains ):
            curU = curUSamples[ i ].tolist()

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
                curX, _ = natafTrans.getX( curU )
                curUSamples[ curIdx ] = curU
                curXSamples[ curIdx ] = curX
                curLsfValues[ curIdx ] = g( curX )
                curIdx += 1

        numSteps += 1
        if lsfLevel <= 0:
            break

    # Save sample and values in the last subset if break 
    if numSteps < maxSubsets:
        allUSamples[ numSteps, :, : ] = curUSamples
        allXSamples[ numSteps, :, : ] = curXSamples
        allLsfValues[ numSteps, : ] = curLsfValues

    pf = np.prod( allProbs[ : numSteps ] )
    return pf, allLsfValues[ : numSteps ], \
        allUSamples[ : numSteps ], allXSamples[ : numSteps ]
