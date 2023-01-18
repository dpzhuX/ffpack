#!/usr/bin/env python3

import numpy as np
from scipy import stats
from ffpack.rpm import metropolisHastings, nataf

def subsetSimulation( dim, g, distObjs, corrMat, numSamples, 
                      numSubsets, probLevel=0.1, quadDeg=99, quadRange=8 ):
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
    numSubsets: scalar
        Number of subsets used to compute the failure probability.
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
    >>> numSamples, numSubsets = 500, 10
    >>> pf = subsetSimulation( dim, g, distObjs, corrMat, numSamples, numSubsets )
    '''
    # Check edge cases

    # Perform Nataf transformation
    natafTrans = nataf.NatafTransformation( distObjs=distObjs, corrMat=corrMat,
                                            quadDeg=quadDeg, quadRange=quadRange )

    numChains = numSamplesRemained = int( probLevel * numSamples )
    # We create each chain for each remained sample
    numSamplesEachChain = int( numSamples / numChains )
    
    numSteps = 0
    # Use curde Monte Carlo to run the first iteration 
    numSteps += 1
    samples = np.random.normal( loc=0, scale=1.0, size=( numSamples, dim ) )
    lsfValues = np.zeros( numSamples )
    for idx, U in enumerate( samples ):
        X, _ = natafTrans.getX( U )
        lsfValues[ idx ] = g( X )
    
    lsfIdx = np.argsort( lsfValues )
    lsfLimit = lsfValues[ lsfIdx[ numSamplesRemained - 1 ] ]

    while lsfLimit > 0 and numSteps < numSubsets:
        numSteps += 1
        curSamples = samples
        nxtSamples = [ ]
        # Sample each chain
        for i in range( numChains ):
            initialVal = curSamples[ lsfIdx[ i ] ].tolist()

            def tpdf( x ):
                return stats.norm().pdf( x )
            
            targetPdf = [ tpdf ] * dim
            
            def pcs( x ):
                return np.random.uniform( x - 0.5, 1 )
            
            proposalCSampler = [ pcs ] * dim 
            
            auMMHSampler = metropolisHastings.\
                AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )
            for _ in range( numSamplesEachChain ):
                nxtSamples.append( auMMHSampler.getSample() )
        curSamples = np.copy( nxtSamples )
        lsfValues = np.zeros( numSamples )
        for idx, U in enumerate( curSamples ):
            X, _ = natafTrans.getX( U )
            lsfValues[ idx ] = g( X )
        
        lsfIdx = np.argsort( lsfValues )
        lsfLimit = lsfValues[ lsfIdx[ numSamplesRemained - 1 ] ]

    pfLastSubset = ( lsfValues < 0 ).sum() / numSamples
    pf = probLevel ** numSteps * pfLastSubset
    print( numSteps )
    print( pf )
    