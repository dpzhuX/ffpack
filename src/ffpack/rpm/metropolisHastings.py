#!/usr/bin/env python3

import numpy as np

class MetropolisHastingsSampler:
    '''
    Metropolis-Hastings sampler to sample data for arbitrary distribution.
    '''
    def __init__( self, initialVal=None, targetPdf=None, proposalCSampler=None, 
                  sampleDomain=lambda X: True ):
        '''
        Initialize the Metropolis-Hastings sampler
        
        Parameters
        ----------
        initialVal: scalar or array_like
            Initial observed data point.
        targetPdf: function
            Target probability density function or target distribution function.
            targetPdf takes one input parameter and return the corresponding 
            probability. It will be called as targetPdf( X ) where X is the same 
            type as initialVal, and a scalar value of probability should be returned.
        proposalCSampler: function
            Proposal conditional sampler (i.e., transition kernel). proposalCSampler 
            is a sampler that will return a sample for the given observed data point. 
            A usual choice is to let proposalCSampler be a Gaussian/normal 
            distribution centered at the observed data point. It will be called as 
            proposalCSampler( X ) where X is the same type as initialVal, and a 
            sample with the same type of initialVal should be returned.
        sampleDomain: function
            Sample domain function. sampleDomain is a function to determine if a
            sample is in the sample domain. For example, it the sample doamin is 
            [ 0, inf ] and the sample is -2, the sample will be rejected. For the 
            sampling on field of real numbers, it should return True regardless of 
            the sample value. It called as sampleDomain( X ) where X is the same 
            type as initivalVal, and a boolean value should be returned.
        
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
        ...                                        proposalCsampler )
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
        self.sampleDomain = sampleDomain
    
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
        if u <= acceptanceRatio and self.sampleDomain( candi ):
            self.nxt = candi
        else:
            self.nxt = self.cur
        self.cur = self.nxt
        return self.cur
