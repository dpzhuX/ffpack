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
    
    # Use curde Monte Carlo to run the first iteration 
    numSteps = 1
    samples = np.random.normal( loc=0, scale=1.0, size=( numSamples, dim  ) )
    lsfValues = np.zeros( numSamples )
    for idx, U in enumerate( samples ):
        X, _ = natafTrans.getX( U )
        lsfValues[ idx ] = g( X )
    
    lsfIdx = np.argsort( lsfValues )

    curSamples = samples[ lsfIdx[ : numSamplesRemained ] ]
    curLsfValues = lsfValues[ lsfIdx[ : numSamplesRemained ] ]
    lsfLevel = curLsfValues[ -1 ]
    if lsfLevel < 0:
        lsfLevel = 0
        probLevel = ( lsfValues < 0 ).sum() / numSamples
    nxtLsfValues = lsfValues

    while lsfLevel >= 0 and numSteps < numSubsets:
        numSteps += 1
        nxtSamples = curSamples
        nxtLsfValues = curLsfValues
        # Sample each chain
        for i in range( numChains ):
            initialVal = curSamples[ i ].tolist()

            def tpdf( x ):
                return stats.norm().pdf( x )
            
            targetPdf = [ tpdf ] * dim
            
            def pcs( x ):
                return np.random.uniform( x - 0.5, x + 0.5 )
            
            proposalCSampler = [ pcs ] * dim 

            def sampleDomainFunc( X, **kwargs ):
                lsfFunc = kwargs[ 'lsfFunc' ]
                lsfLevel = kwargs[ 'lsfLevel' ]
                return lsfFunc( X ) < lsfLevel
            
            auMMHSampler = metropolisHastings.\
                AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler,
                                     sampleDomain=sampleDomainFunc,
                                     lsfFunc=g,
                                     lsfLevel=lsfLevel )
            for _ in range( numSamplesEachChain - 1 ):
                nxtU = auMMHSampler.getSample()
                nxtX, _ = natafTrans.getX( nxtU )
                nxtSamples = np.append( nxtSamples, np.array( [ nxtU ] ), axis=0 )
                nxtLsfValues = np.append( nxtLsfValues, g( nxtX ) )
        
        lsfIdx = np.argsort( nxtLsfValues )
        curSamples = nxtSamples[ lsfIdx[ : numSamplesRemained ] ]
        curLsfValues = nxtLsfValues[ lsfIdx[ : numSamplesRemained ] ]
        lsfLevel = curLsfValues[ -1 ]

    pfLastSubset = ( nxtLsfValues < 0 ).sum() / numSamples
    print(( nxtLsfValues < 0).sum() )
    pf = probLevel ** ( numSteps - 1 ) * pfLastSubset
    print( f"{pf:.8f}" )
