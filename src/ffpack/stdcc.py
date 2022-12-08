#!/usr/bin/env python3

'''
This module implement the standard cycle counting methods in 
ASTM E1049-85(2017) Standard Practices for Cycle Counting in Fatigue Analysis
'''

import numpy as np
from collections import defaultdict

def levelCrossingCounting( data, levels=None ):
    '''
    Implement the level crossing counting method based on E1049-85: sec 5.1.1
    By default, this method does the level crossing couting for each integers
    Args:
        data: 1D input sequence data for couning
        leves: 1D input sequence levels

    Returns:
        rst: 2D sorted output data

    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if not levels:
        minElement = np.floor( np.min( data ) )
        maxElement = np.ceil( np.max( data ) )
        numElement = maxElement - minElement + 1
        levels =  np.linspace( minElement, maxElement, numElement.astype(int) )
    else:
        levels = np.array( sorted( set( levels ) ) )

    rstDict = defaultdict( int )
    # Check each interval
    for i in range( len( data ) - 1 ):
        intervalStart = data[ i ]
        intervalEnd = data[ i + 1 ]
        lowerVal = intervalStart if intervalStart <= intervalEnd else intervalEnd
        upperVal = intervalEnd if intervalStart <= intervalEnd else intervalStart
        leftIndex = np.searchsorted( levels, lowerVal, side='left' )
        if i != 0 and leftIndex < len( levels ) and lowerVal == levels[ leftIndex ]:
            leftIndex += 1
        rightIndex = np.searchsorted( levels, upperVal, side='right' )
        for j in range( leftIndex, rightIndex ):
            if ( intervalStart <= intervalEnd and levels[ j ] >= 0 ) or \
               ( intervalStart > intervalEnd and levels[ j ] < 0 ):
                rstDict[ levels[ j ] ] += 1
    if len( rstDict ) == 0:
        return np.array( [ [] ] )
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst

def peakCounting( data ):
    pass

def simpleRangeCounting( data ):
    pass

def rainflowCounting( data ):
    pass

def addOne( num ):
    return num + 1