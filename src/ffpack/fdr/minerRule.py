#!/usr/bin/env python3

'''
Palmgren-Miner damage rule is one of the famous fatigue damage rule for 
fatigue estimation. The rule is defined in a simple and intuitive way and
it is very popular now.

Reference: Miner, M.A., 1945. Cumulative damage in fatigue.
'''

import numpy as np
from ffpack import utils

def minerDamageRuleNaive( fatigueData ):
    '''
    Naive Palmgren-miner damage rule directly calcuates the damage results.

    Parameters
    ----------
    fatigueData: 2d array 
        Paired counting and experimental data under each load condition,
        e.g., [ [ C_1, F_1 ], [ C_2, F_2 ], ..., [ C_i, F_i ] ] 
        where C_i and F_i represent the counting cycles and failure cycles
        under the same load condition.
    
    Returns
    -------
    rst: scalar
        Fatigue damage calculated based on the Palmgren-miner rule
    
    Raises
    ------
    ValueError
        If fatigueData length is less than 1.
        If counting cycles is less than 0.
        If failure cycles is less than or equal 0.
        If counting cycles is large than failure cycles.

    
    Examples
    --------
    >>> from ffpack.fdr import minerDamageRuleNaive
    >>> fatigueData = [ [ 10, 100 ], [ 200, 2000 ] ]
    >>> rst = minerDamageRuleNaive( fatigueData )
    '''
    # Edge case check
    fatigueData = np.array( fatigueData )
    if len( fatigueData.shape ) != 2:
        raise ValueError( "Input fatigueData dimension should be 2" )
    if fatigueData.shape[ 0 ] < 1:
        raise ValueError( "Input data length should be at least 1" )
    for p in fatigueData:
        if len( p ) != 2:
            raise ValueError( "Each pair length in fatigueData should be 2" )
        if p[ 0 ] < 0:
            raise ValueError( "Counting cycles should be larger than or equal 0" )
        if p[ 1 ] <= 0:
            raise ValueError( "Failure cycles should be larger than 0" )
        if p[ 0 ] > p[ 1 ]:
            raise ValueError( "Failure cycles should be larger than or equal counting cycles" )

    return np.sum( fatigueData[ :, 0 ] / fatigueData[ :, 1 ] )


def minerDamageRuleClassic( lccData, snData, fatigueLimit ):
    '''
    Classical Palmgren-miner damage rule calcuates the damage results based on SN curve.
    
    Parameters
    ----------
    lccData: 2d array
        Load cycle counting results in 2D matrix,
        e.g., [ [ range, counts ], ... ]
    
    snData: 2d array
        Experimental SN data in 2D matrix,
        e.g., [ [ N_1, S_1 ], [ N_2, S_2 ], ..., [ N_i, S_i ] ]
    
    fatigueLimit: scalar
        Fatigue limit indicating the minimum S that can cause fatigue.
    
    Returns
    -------
    rst: scalar
        Fatigue damage calculated based on the Palmgren-miner rule
    
    Raises
    ------
    ValueError
        If the lccData dimension is not 2.
        If the lccData length is less than 1.

    Examples
    --------
    >>> from ffpack.fdr import minerDamageRuleClassic
    >>> lccData = [ [ 1, 100 ], [ 2, 10 ] ]
    >>> snData = [ [ 10, 3 ], [ 1000, 1 ] ]
    >>> fatigueLimit = 0.5
    >>> rst = minerDamageRuleClassic( lccData, snData, fatigueLimit )
    '''
    # Edge case check
    lccData = np.array( lccData )
    if len( lccData.shape ) != 2:
        raise ValueError( "Input lccData dimension should be 2" )
    if lccData.shape[ 0 ] < 1:
        raise ValueError( "Input lccData length should be at least 1" )
    for p in lccData:
        if len( p ) != 2:
            raise ValueError( "Each pair length in lccData should be 2" )
        if p[ 0 ] <= 0:
            raise ValueError( "Range should be larger than 0" )
        if p[ 1 ] <= 0:
            raise ValueError( "Counts should be larger than 0" )
    
    snCurveFitter = utils.SnCurveFitter( snData, fatigueLimit=fatigueLimit )

    rst = 0
    for p in lccData:
        if snCurveFitter.getN( p[ 0 ] ) != -1: 
            rst += p[ 1 ] / snCurveFitter.getN( p[ 0 ] )

    return rst 
