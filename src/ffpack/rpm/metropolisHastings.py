#!/usr/bin/env python3

import numpy as np

class MetropolisHastingsSampler:
    '''
    Metropolis-Hastings sampler to sample data for arbitrary distribution.
    '''
    def __init__( self, initialVal=None, targetPdf=None, proposalCSampler=None ):
        '''
        Initialize the Metropolis-Hastings sampler
        
        Parameters
        ----------
        initialVal: scalar or array_like of scalar
            Initial observed data point.
        targetPdf: function
            Target probability density function or target distribution function.
            targetPdf takes one input parameter and return the corresponding 
            probability.
        proposalCSampler: function
            Proposal conditional sampler. proposalCSampler is a sampler that will 
            return a sample for the given observed data point. A usual choice is to 
            let proposalCSampler be a Gaussian/normal distribution centered at the 
            observed data point.
        
        Raises
        ------
        ValueError
            If any input parameter is None.
            If targetPdf returns negative value.

        Examples
        --------
        >>> from ffpack.rpm import MetropolisHastingsSampler
        >>> initialVal = 1.0
        >>> targetPdf = lambda x : 0 if x < 0 else np.exp( -x )
        >>> proposalCSampler = lambda x : np.random.normal( x, 1 )
        >>> mhSampler = MetropolisHastingsSampler( initialVal, targetPdf, 
        >>>                                        proposalCsampler )
        '''
        if initialVal is None:
            raise ValueError( "initialVal cannot be None" )
        if targetPdf is None:
            raise ValueError( "targetPdf cannot be None" )
        if proposalCSampler is None:
            raise ValueError( "proposalCSampler cannot be None" )
        self.cur = initialVal
        self.nxt = initialVal
        self.targetPdf = targetPdf
        self.proposalCSampler = proposalCSampler
    
    def getAcceptanceRatio( self, candi ):
        fcur = self.targetPdf( self.cur )
        fcandi = self.targetPdf( candi )
        if fcur < 0 or fcandi < 0:
            raise ValueError( "targetPdf cannot return negative value" )
        return fcandi / fcur
    
    def getCandidate( self ):
        return self.proposalCSampler( self.cur )

    def getSample( self ):
        '''
        Get a sample.

        Returns
        -------
        rst: scalar or array_like of scalar
            Data point sample.
        
        Examples
        --------
        >>> mhSampler.getSample()
        '''
        candi = self.getCandidate()
        acceptanceRatio = self.getAcceptanceRatio( candi )
        u = np.random.uniform()
        if u <= acceptanceRatio:
            self.nxt = candi
        else:
            self.nxt = self.cur
        self.cur = self.nxt
        return self.cur
