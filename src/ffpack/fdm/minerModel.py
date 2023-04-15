#!/usr/bin/env python3

'''
Palmgren-Miner damage model is one of the famous fatigue damage models for 
fatigue estimation. The model is defined in a simple and intuitive way and
it is very popular now.

Reference: Miner, M.A., 1945. Cumulative damage in fatigue.
'''

import numpy as np
from ffpack import utils


def minerDamageModelNaive( fatigueData ):
    '''
    Naive Palmgren-miner damage model directly calcuates the damage results.

    Parameters
    ----------
    fatigueData: 2d array 
        Paired counting and experimental data under each load condition,
        e.g., [ [ C1, F1 ], [ C2, F2 ], ..., [ Ci, Fi ] ] 
        where Ci and Fi represent the counting cycles and failure cycles
        under the same load condition.
    
    Returns
    -------
    rst: scalar
        Fatigue damage calculated based on the Palmgren-miner model
    
    Raises
    ------
    ValueError
        If fatigueData length is less than 1.
        If counting cycles is less than 0.
        If number of failure cycles is less than or equal 0.
        If number of counting cycles is large than failure cycles.

    
    Examples
    --------
    >>> from ffpack.fdm import minerDamageModelNaive
    >>> fatigueData = [ [ 10, 100 ], [ 200, 2000 ] ]
    >>> rst = minerDamageModelNaive( fatigueData )
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
            raise ValueError( "Failure cycles should be larger than "
                              "or equal counting cycles" )

    return np.sum( fatigueData[ :, 0 ] / fatigueData[ :, 1 ] )


def minerDamageModelClassic( lccData, snData, fatigueLimit ):
    '''
    Classical Palmgren-miner damage model calculates the damage results 
    based on the SN curve.
    
    Parameters
    ----------
    lccData: 2d array
        Load cycle counting results in a 2D matrix,
        e.g., [ [ value, count ], ... ]
    
    snData: 2d array
        Experimental SN data in 2D matrix,
        e.g., [ [ N1, S1 ], [ N2, S2 ], ..., [ Ni, Si ] ]
    
    fatigueLimit: scalar
        Fatigue limit indicating the minimum S that can cause fatigue.
    
    Returns
    -------
    rst: scalar
        Fatigue damage calculated based on the Palmgren-miner model.
    
    Raises
    ------
    ValueError
        If the lccData dimension is not 2.
        If the lccData length is less than 1.

    Examples
    --------
    >>> from ffpack.fdr import minerDamageModelClassic
    >>> lccData = [ [ 1, 100 ], [ 2, 10 ] ]
    >>> snData = [ [ 10, 3 ], [ 1000, 1 ] ]
    >>> fatigueLimit = 0.5
    >>> rst = minerDamageModelClassic( lccData, snData, fatigueLimit )
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
        nFromSNCurve = snCurveFitter.getN( p[ 0 ] )
        if nFromSNCurve != -1: 
            rst += p[ 1 ] / nFromSNCurve

    return rst 
