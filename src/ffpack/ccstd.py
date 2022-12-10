#!/usr/bin/env python3

'''
This module implements the standard cycle counting methods in 
ASTM E1049-85(2017) Standard Practices for Cycle Counting in Fatigue Analysis
'''

import numpy as np
from ffpack import utils
from collections import defaultdict, deque

def levelCrossingCounting( data, levels=None ):
    '''
    Implement the level crossing counting method based on E1049-85: sec 5.1.1
    By default, this method does the level crossing couting for each integers
    Args:
        data: 1D input sequence data for couning
        levels: 1D input sequence levels

    Returns:
        rst: 2D sorted output data
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if levels is None or len( levels )==0 :
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
        return [ [] ]
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()

def peakCounting( data, level=None ):
    '''
    Implement the peak counting method based on E1049-85: sec 5.2.1
    By default, this method does the peak crossing couting for y == 0
    Args:
        data: 1D input sequence data for couning
        level: Scalar data

    Returns:
        rst: 2D sorted output data
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if level is None:
        level =  0.0;
    
    rstDict = defaultdict( int )
    for i, cur in enumerate( data ):
        if i == 0 or i == len( data ) - 1:
            continue
        
        # Compare the prev and next
        prev = data[ i - 1 ]
        next = data[ i + 1 ]
        if ( prev < cur and cur > next and cur >= level ) or \
           ( prev > cur and cur < next and cur < level ):
            rstDict[ cur ] += 1

    if len( rstDict ) == 0:
        return [ [] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()

def simpleRangeCounting( data ):
    '''
    Implement the simple range counting method based on E1049-85: sec 5.3.1
    Args:
        data: 1D input sequence data for couning

    Returns:
        rst: 2D sorted output data
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    # Remove the intermediate value first
    data = np.array( utils.getPeakAndValley( data, keepEnds=True ) )

    rstDict = defaultdict( int )
    for i, cur in enumerate( data ):
        if i == 0:
            continue
        prev = data[ i - 1 ]
        rstDict[ abs( prev - cur ) ] += 1

    if len( rstDict ) == 0:
        return [ [] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()

def rainflowCounting( data ):
    '''
    Implement the rainflow counting method based on E1049-85: sec 5.4.4
    Args:
        data: 1D input sequence data for couning

    Returns:
        rst: 2D sorted output data
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    # Remove the intermediate value first
    data = np.array( utils.getPeakAndValley( data, keepEnds=True ) )

    dequeA = deque()
    dequeB = deque( [ i for i in data ] )
    S = None
    YContainsS = None
    rstDict = defaultdict( int )
    while len( dequeB ) >= 3:
        A = dequeB.popleft()
        B = dequeB.popleft()
        C = dequeB.popleft()
        Y = abs( A - B )
        X = abs( B - C )
        if S is None:
            S = A
            YContainsS = True
        if X >= Y:
            if YContainsS:
                rstDict[ Y ] += 0.5
                dequeB.appendleft( C )
                dequeB.appendleft( B )
                S = None
                YContainsS = None
            else:
                rstDict[ Y ] += 1
                dequeB.appendleft( C )
                while dequeA:
                    dequeB.appendleft( dequeA.pop() )
                S = None
                YContainsS = None
        else:
            YContainsS = False
            dequeB.appendleft( C )
            dequeB.appendleft( B )
            dequeA.append( A )

    while dequeA:
        dequeB.appendleft( dequeA.pop() )
    
    A = dequeB.popleft()
    while dequeB:
        B = dequeB.popleft()
        Y = abs( A - B )
        rstDict[ Y ] += 0.5
        A = B

    if len( rstDict ) == 0:
        return [ [] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
