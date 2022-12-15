#!/usr/bin/env python3

import numpy as np

class SnCurveFitter:
    '''
    Fit a SN curve based on the experimental data.
    
    Parameters
    ----------
    data: 2d array
        Experimental data for fitting in 2D matrix,
        e.g., [ [ N_1, S_1 ], [ N_2, S_2 ], ..., [ N_i, S_i ] ]
    
    fatigueLimit: scalar
        Fatigue limit indicating the minimum S that can cause fatigue.
    
    Raises
    ------
    ValueError
        If the data dimension is not 2.
        If the data length is less than 2.
        If the fatigueLimit is less than or equal 0.
        If N_i or S_i is less than or equal 0.

    Examples
    --------
    >>> from ffpack.utils import snCurveFitter
    >>> data = [ [ 10, 3 ], [ 1000, 1 ] ]
    >>> fatigueLimit = 0.5
    >>> snCurveFitter = SnCurveFitter( data, fatigueLimit )
    '''
    def __init__( self, data, fatigueLimit ):
        # Edge case check
        data = np.array( data )
        if len( data.shape ) != 2:
            raise ValueError( "Input data dimension should be 2" )
        if data.shape[0] < 2:
            raise ValueError( "Input data length should be at least 2" )
        if fatigueLimit <= 0:
            raise ValueError( "fatigueLimit should be larger than 0" )
        for p in data:
            if p[ 0 ] <= 0 or p[ 1 ] <= 0:
                raise ValueError( "S_i and N_i should be larger than 0" )
        
        self.fatigueLimit = fatigueLimit

        logN = np.log10( data.T[ 0 ] )
        S = data.T[ 1 ]

        coef = np.polyfit( S, logN, 1 )
        self.fitter = np.poly1d( coef )

    def getN( self, S ):
        '''
        Query fatigue life N for a given S

        Parameters
        ----------
        S: scalar
            Input S for fatigue life query.
        
        Returns
        -------
        rst: scalar
            Fatigue life under the query S. 
            If S is less than or equal fatigueLimit, -1 will be returned. 

        Raises
        ------
        ValueError
            If the S is less than or equal 0.
        
        Examples
        --------
        >>> rst = snCurveFitter.getN( 2 )
        '''
        if S <= 0:
            raise ValueError( "S should be larger than 0" )

        if S <= self.fatigueLimit:
            return -1
        else:
            return np.power( 10, self.fitter( S ) )
