#!/usr/bin/env python3

'''
This module implements the standard cycle counting methods in 
ASTM E1049-85(2017) Standard Practices for Cycle Counting in Fatigue Analysis
'''

import numpy as np
from ffpack import utils
from collections import defaultdict, deque

def astmLevelCrossingCounting( data, refLevel=0.0, levels=None ):
    '''
    Implement the level crossing counting method based on E1049-85: sec 5.1.1
    By default, this method does the level crossing couting for each integers
    at the reference level of 0.0.

    Args:
        data: 1D sequence data for couning
        refLevel: scalar value indicating the reference level
        levels: 1D sequence of self-defined levels

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
            if ( intervalStart <= intervalEnd and levels[ j ] >= refLevel ) or \
               ( intervalStart > intervalEnd and levels[ j ] < refLevel ):
                rstDict[ levels[ j ] ] += 1
    if len( rstDict ) == 0:
        return [ [] ]
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()

def astmPeakCounting( data, refLevel=None ):
    '''
    Implement the peak counting method based on E1049-85: sec 5.2.1
    By default, this method does the peak crossing couting for y == 0
    Args:
        data: 1D input sequence data for couning
        refLevel: Scalar data

    Returns:
        rst: 2D sorted output data
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if refLevel is None:
        refLevel =  0.0;
    
    rstDict = defaultdict( int )
    for i, cur in enumerate( data ):
        if i == 0 or i == len( data ) - 1:
            continue
        
        # Compare the prev and next
        prev = data[ i - 1 ]
        next = data[ i + 1 ]
        if ( prev < cur and cur > next and cur >= refLevel ) or \
           ( prev > cur and cur < next and cur < refLevel ):
            rstDict[ cur ] += 1

    if len( rstDict ) == 0:
        return [ [] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()

def astmSimpleRangeCounting( data ):
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
    data = np.array( utils.getSequencePeakAndValleys( data, keepEnds=True ) )

    rstDict = defaultdict( int )
    for i, cur in enumerate( data ):
        if i == 0:
            continue
        prev = data[ i - 1 ]
        rstDict[ abs( prev - cur ) ] += 0.5

    if len( rstDict ) == 0:
        return [ [] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()

def astmRainflowCounting( data ):
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
    data = np.array( utils.getSequencePeakAndValleys( data, keepEnds=True ) )

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


def rychlikRainflowCycleCounting( data, aggragate=True ):
    '''
    Implement the rainflow counting method based on Definition 1
    in "A new definition of the rainflow cycle counting method" by Rychilk on IJF
    Args:
        data: 1D input sequence data for couning

    Returns:
        rst: 1D range H(t) sequence 
    '''

    def getMinLeft( data, i ):
        if ( i == 1 ): 
            return min( data[ 0 ], data[ 1 ] )
        
        left = data[ i - 1 ]
        j = i - 2
        while ( j >= 0 and data[ j ] < data[ i ] ): 
            left = min( left, data[ j ] )
            j -= 1

        return left

    def getMinRight( data, i ):
        if ( i == len( data ) - 2 ): 
            return min( data[ len( data ) - 1 ], data[ len( data ) - 2 ] )
        
        right = data[ i + 1 ]
        j = i + 2
        while ( j < len( data ) and data[ j ] < data[ i ] ):
            right = min( right, data[ j ] )
            j += 1

        return right 


    peakVallays = utils.getSequencePeakAndValleys( data, keepEnds=True )
    rstSeq = [ ]
    for i in range( 1, len( data ) - 1 ):
        if ( data[ i ] > data[ i - 1 ] and data[ i ] > data[ i + 1 ] ):
            # TODO: set a global variable for round digit
            height = data[ i ] - max( getMinLeft( data, i ), getMinRight( data, i ))
            rstSeq.append( round( height, 7 ) )
    
    if ( not aggragate ): 
        return rstSeq

    rstDict = defaultdict( int )
    for i, cur in enumerate( rstSeq ):
        rstDict[ cur ] += 1

    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
