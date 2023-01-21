#!/usr/bin/env python3

import numpy as np

class MetropolisHastingsSampler:
    '''
    Metropolis-Hastings sampler to sample data [Bourinet2018]_.


    References
    ----------
    .. [Bourinet2018] Bourinet, J.M., 2018. Reliability analysis and optimal design 
       under uncertainty-Focus on adaptive surrogate-based approaches (Doctoral 
       dissertation, Université Clermont Auvergne).
    '''
    def __init__( self, initialVal=None, targetPdf=None, proposalCSampler=None, 
                  sampleDomain=None, **sampleDomainKwargs ):
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
        sampleDomain: function, optional
            Sample domain function. sampleDomain is a function to determine if a
            sample is in the sample domain. For example, it the sample doamin is 
            [ 0, inf ] and the sample is -2, the sample will be rejected. For the 
            sampling on field of real numbers, it should return True regardless of 
            the sample value. It called as sampleDomain( X ) where X is the same 
            type as initivalVal, and a boolean value should be returned.
        
        Raises
        ------
        ValueError
            If initialVal, targetPdf, or proposalCSampler is None.
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
        self.sampleDomainKwargs = sampleDomainKwargs
        if sampleDomain is None:
            sampleDomain = lambda cur, nxt, **sampleDomainKwargs: True
        self.cur = np.copy( initialVal )
        self.nxt = np.copy( initialVal )
        self.isArray = len( self.cur.shape ) != 0
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
        if u <= acceptanceRatio:
            if self.isArray:
                self.nxt[ : ] = candi[ : ]
            else:
                self.nxt = candi
        else:
            if self.isArray:
                self.nxt[ : ] = self.cur[ : ]
            else:
                self.nxt = self.cur
        if self.sampleDomain( self.cur, self.nxt, **self.sampleDomainKwargs ):
            if self.isArray:
                self.cur[ : ] = self.nxt[ : ]
            else:
                self.cur = self.nxt
        return self.cur


class AuModifiedMHSampler:
    '''
    Modified Metropolis-Hastings sampler based on Au and Beck algorithm [Au2001]_.
    
    References
    ----------
    .. [Au2001] Au, S.K. and Beck, J.L., 2001. Estimation of small failure 
       probabilities in high dimensions by subset simulation. Probabilistic 
       engineering mechanics, 16(4), pp.263-277.
    '''
    def __init__( self, initialVal=None, targetPdf=None, proposalCSampler=None, 
                  sampleDomain=None, **sampleDomainKwargs ):
        '''
        Initialize the Au modified Metropolis-Hastings sampler
        
        Parameters
        ----------
        initialVal: array_like
            Initial observed data point.
        targetPdf: function list
            Target probability density function list. Each element targetPdf[ i ] in 
            the list is a callable function referring the independent marginal.
            targetPdf[ i ] takes one input parameter and return the corresponding 
            probability. It will be called as targetPdf[ i ]( X[ i ] ) where X is a 
            list in which the element is same type as initialVal[ i ], and a scalar 
            value of probability should be returned by targetPdf[ i ]( X[ i ] ).
        proposalCSampler: function list
            Proposal conditional sampler list (i.e., transition kernel list). Each 
            element proposalCSampler[ i ] in the list is a callable function
            referring a sampler that will return a sample for the given observed 
            data point.  A usual choice is to let proposalCSampler[ i ] be a 
            Gaussian/normal distribution centered at the observed data point. 
            It will be called as proposalCSampler[ i ]( X[ i ] ) where X is a list 
            in which each element is the same type as initialVal[ i ], and a 
            sample with the same type of initialVal[ i ] should be returned.
        sampleDomain: function, optional
            Sample domain function. sampleDomain is a function to determine if a
            sample is in the sample domain. For example, it the sample doamin is 
            [ 0, inf ] and the sample is -2, the sample will be rejected. For the 
            sampling on field of real numbers, it should return True regardless of 
            the sample value. It called as sampleDomain( X, **sampleDomainKwargs ) 
            where X is a list in which each element is the same type as 
            initivalVal[ i ], and a boolean value should be returned.
        
        Raises
        ------
        ValueError
            If initialVal, targetPdf, or proposalCSampler is None.
            If dims of initialVal, targetPdf, and proposalCSampler are not equal.
            If targetPdf returns negative value.

        Examples
        --------
        >>> from ffpack.rpm import AuModifiedMHSampler
        >>> initialValList = [ 1.0, 1.0 ]
        >>> targetPdf = [ lambda x : 0 if x < 0 else np.exp( -x ),
        ...               lambda x : 0 if x < 0 else np.exp( -x ) ]
        >>> proposalCSampler = [ lambda x : np.random.normal( x, 1 ),
        ...                      lambda x : np.random.normal( x, 1 ) ]
        >>> auMMHSampler = AuModifiedMHSampler( initialVal, targetPdf, 
        ...                                     proposalCsampler )
        '''
        if initialVal is None or not isinstance( initialVal, list ):
            raise ValueError( "initialVal should be a list of initival values." )
        if targetPdf is None or not isinstance( targetPdf, list ):
            raise ValueError( "targetPdf should be a list of target marginal "
                              "distribution functions." )
        if proposalCSampler is None or not isinstance( proposalCSampler, list ):
            raise ValueError( "proposalCSampler should be a list of "
                              "conditional samplers." )
        self.sampleDomainKwargs = sampleDomainKwargs
        if sampleDomain is None:
            sampleDomain = lambda cur, nxt, **sampleDomainKwargs: True
        self.dim = len( initialVal )
        if self.dim != len( targetPdf ) or self.dim != len( proposalCSampler ):
            raise ValueError( "dimensions of initialVal, targetPdf, and "
                              "proposalCSampler should be equal." )
        self.cur = np.copy( initialVal )
        self.nxt = np.copy( initialVal )
        self.targetPdf = targetPdf
        self.proposalCSampler = proposalCSampler
        self.sampleDomain = sampleDomain
    
    def getAcceptanceRatio( self, candi, i ):
        fcur = self.targetPdf[ i ]( self.cur[ i ] )
        fcandi = self.targetPdf[ i ]( candi )
        if fcur < 0 or fcandi < 0:
            raise ValueError( f"targetPdf[ { i } ] cannot return negative value" )
        return fcandi / fcur
    
    def getCandidate( self, i ):
        return self.proposalCSampler[ i ]( self.cur[ i ] )

    def getSample( self ):
        '''
        Get a sample.

        Returns
        -------
        rst: array_like
            Data point sample.
        
        Examples
        --------
        >>> auMMHSampler.getSample()
        '''
        for i in range( self.dim ):
            candi = self.getCandidate( i )
            acceptanceRatio = self.getAcceptanceRatio( candi, i )
            u = np.random.uniform()
            if u <= acceptanceRatio:
                self.nxt[ i ] = candi
            else:
                self.nxt[ i ] = self.cur[ i ]
        if self.sampleDomain( self.cur, self.nxt, **self.sampleDomainKwargs ):
            self.cur[ : ] = self.nxt[ : ]
        return self.cur
