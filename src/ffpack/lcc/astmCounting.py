#!/usr/bin/env python3

'''
This module implements the standard cycle counting methods in 
ASTM E1049-85(2017) Standard Practices for Cycle Counting in Fatigue Analysis
'''

import numpy as np
from ffpack.utils import generalUtils
from ffpack.config import globalConfig
from collections import defaultdict, deque

def astmLevelCrossingCounting( data, refLevel=0.0, levels=None, aggregate=True ):
    '''
    ASTM level crossing counting in E1049-85: sec 5.1.1.

    Parameters
    ----------
    data: 1d array
        Load sequence data for counting.
    refLevel: scalar, optional
        Reference level.
    levels: 1d array
        Self-defined levels for counting.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting,
        e.g., [ crossPoint1, corssPoint2, ... ], will be returned.

    Returns
    -------
    rst: 2d array
        Sorted counting results.
    
    Raises
    ------
    ValueError
        If the data length is less than 2 or the data dimension is not 1.

    Examples
    --------
    >>> from ffpack.lcc import astmLevelCrossingCounting
    >>> data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
    >>>          -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    >>> rst = astmLevelCrossingCounting( data )
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if levels is None or len( levels ) == 0:
        minElement = np.floor( np.min( data ) )
        maxElement = np.ceil( np.max( data ) )
        numElement = maxElement - minElement + 1
        levels = np.linspace( minElement, maxElement, numElement.astype(int) )
    else:
        levels = np.array( sorted( set( levels ) ) )

    # Remove the intermediate value first
    data = np.array( generalUtils.sequencePeakAndValleys( data, keepEnds=True ) )

    rstDict = defaultdict( int )
    rstSeq = [ ]
    # Check each interval
    # Each interval will be searched from small to large value
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
                rstSeq.append( levels[ j ] )
    if len( rstDict ) == 0:
        return [ [ ] ] if aggregate else [ ]
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist() if aggregate else rstSeq


def astmPeakCounting( data, refLevel=None, aggregate=True ):
    '''
    ASTM peak counting in E1049-85: sec 5.2.1.

    Parameters
    ----------
    data: 1d array
        Load sequence data for counting.
    refLevel: scalar, optional
        Reference level.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting,
        e.g., [ peak1, peak2, ... ], will be returned.
    
    Returns
    -------
    rst: 2d array
        Sorted counting results.
    
    Raises
    ------
    ValueError
        If the data length is less than 2 or the data dimension is not 1.

    Examples
    --------
    >>> from ffpack.lcc import astmPeakCounting
    >>> data = [ 0.0, 1.5, 0.5, 3.5, 0.5, 2.5, -1.5, -0.5, -2.5, 
    >>>          -2.0, -2.7, -2.5, -3.5, 1.5, 0.5, 3.5, -0.5 ]
    >>> rst = astmPeakCounting( data )
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if refLevel is None:
        refLevel = 0.0
    
    rstDict = defaultdict( int )
    rstSeq = [ ]
    for i, cur in enumerate( data ):
        if i == 0 or i == len( data ) - 1:
            continue
        
        # Compare the prev and next
        prev = data[ i - 1 ]
        next = data[ i + 1 ]
        if ( prev < cur and cur > next and cur >= refLevel ) or \
           ( prev > cur and cur < next and cur < refLevel ):
            rstDict[ cur ] += 1
            rstSeq.append( cur )

    if len( rstDict ) == 0:
        return [ [ ] ] if aggregate else [ ]
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist() if aggregate else rstSeq


def astmSimpleRangeCounting( data, aggregate=True ):
    '''
    ASTM simple range counting in E1049-85: sec 5.3.1.

    Parameters
    ----------
    data: 1d array
        Load sequence data for counting.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting,
        e.g., [ [ rangeStart1, rangeEnd1 ], [ rangeStart2, rangeEnd2 ], ... ], 
        will be returned.
    
    Returns
    -------
    rst: 2d array
        Sorted counting results.
    
    Raises
    ------
    ValueError
        If the data length is less than 2 or the data dimension is not 1.

    Examples
    --------
    >>> from ffpack.lcc import astmSimpleRangeCounting
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmSimpleRangeCounting( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    # Remove the intermediate value first
    data = np.array( generalUtils.sequencePeakAndValleys( data, keepEnds=True ) )

    rstDict = defaultdict( int )
    rstSeq = [ ]
    for i, cur in enumerate( data ):
        if i == 0:
            continue
        prev = data[ i - 1 ]
        rstDict[ abs( prev - cur ) ] += 0.5
        rstSeq.append( [ prev, cur ] )

    if len( rstDict ) == 0:
        return [ [] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist() if aggregate else rstSeq


def astmRainflowCounting( data, aggregate=True ):
    '''
    ASTM rainflow counting in E1049-85: sec 5.4.4.

    Parameters
    ----------
    data: 1d array
        Load sequence data for counting.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting, 
        e.g., [ [ rangeStart1, rangeEnd1, count1 ], 
        [ rangeStart2, rangeEnd2, count2 ], ... ], will be returned.
    
    Returns
    -------
    rst: 2d array
        Sorted counting results.
    
    Raises
    ------
    ValueError
        If the data length is less than 2 or the data dimension is not 1.

    Examples
    --------
    >>> from ffpack.lcc import astmRainflowCounting
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmRainflowCounting( data )
    '''
    # Edge case check
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    # Remove the intermediate value first
    data = np.array( generalUtils.sequencePeakAndValleys( data, keepEnds=True ) )

    dequeA = deque()
    dequeB = deque( [ i for i in data ] )
    S = None
    YContainsS = None
    rstDict = defaultdict( int )
    rstSeq = [ ]
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
                rstSeq.append( [ A, B, 0.5 ])
                dequeB.appendleft( C )
                dequeB.appendleft( B )
                S = None
                YContainsS = None
            else:
                rstDict[ Y ] += 1
                rstSeq.append( [ A, B, 1 ] )
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
        rstSeq.append( [ A, B, 0.5 ] )
        A = B

    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist() if aggregate else rstSeq


def astmRangePairCounting( data, aggregate=True ):
    '''
    ASTM range pair counting in E1049-85: sec 5.4.3.

    Parameters
    ----------
    data: 1d array
        Load sequence data for counting.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting,
        e.g., [ [ rangeStart1, rangeEnd1 ], [ rangeStart2, rangeEnd2 ], ... ], 
        will be returned.
    
    Returns
    -------
    rst: 2d array
        Sorted counting results.
    
    Raises
    ------
    ValueError
        If the data length is less than 2 or the data dimension is not 1.

    Examples
    --------
    >>> from ffpack.lcc import astmRangePairCounting
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmRangePairCounting( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    # Remove the intermediate value first
    data = np.array( generalUtils.sequencePeakAndValleys( data, keepEnds=True ) )
    indices = np.array( range( -1, len( data ) - 1 ) )

    def checkPreviousThree( indices, i ):
        if ( indices[ i ] < 0 ):
            return False
        if ( indices[ indices[ i ] ] < 0 ):
            return False
        return True
    
    rstSeq = [ ]
    # loop from left to right
    i = 2
    while i < len( data ):
        if ( not checkPreviousThree( indices, i ) ): 
            i += 1
            continue
        second = indices[ i ]
        first = indices[ second ]
        left = abs( data[ first ] - data[ second ] )
        right = abs( data[ second ] - data[ i ] )
        if ( left <= right ):
            rstSeq.append( [ data[ first ], data[ second ] ] )
            indices[ i ] = indices[ first ]
            indices[ second ] = -2
            indices[ first ] = -2
        else: 
            i += 1
    
    # loop from right to left
    i = len( data ) - 1
    while i > 1:
        if ( not checkPreviousThree( indices, i ) ): 
            i -= 1
            continue
        second = indices[ i ]
        first = indices[ second ]
        left = abs( data[ first ] - data[ second ] )
        right = abs( data[ second ] - data[ i ] )
        if ( right <= left ):
            rstSeq.append( [ data[ second ], data[ i ] ] )
            indices[ first ] = indices[ i ]
            indices[ second ] = -2
            indices[ i ] = -2
        else: 
            i -= 1

    if ( not aggregate ): 
        return rstSeq
    
    rstDict = defaultdict( int )
    for leftRight in rstSeq:
        height = round( abs( leftRight[ 1 ] - leftRight[ 0 ] ), globalConfig.atol )
        rstDict[ height ] += 1

    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()


def astmRainflowRepeatHistoryCounting( data, aggregate=True ):
    '''
    ASTM simplified rainflow counting for repeating histories in E1049-85: sec 5.4.5.

    Parameters
    ----------
    data: 1d array
        Load sequence data for counting.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting,
        e.g., [ [ rangeStart1, rangeEnd1 ], [ rangeStart2, rangeEnd2 ], ... ], 
        will be returned.
    
    Returns
    -------
    rst: 2d array
        Sorted counting results.
    
    Raises
    ------
    ValueError
        If the data length is less than 2 or the data dimension is not 1.
        If the data is not repeatable: first data point is different from the last data point. 

    Examples
    --------
    >>> from ffpack.lcc import astmRainflowRepeatHistoryCounting
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmRainflowRepeatHistoryCounting( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[ 0 ] <= 1:
        raise ValueError( "Input data length should be at least 2")
    if data[ 0 ] != data[ len( data ) - 1 ]:
        raise ValueError( "Input data should be repeating")

    # Remove the intermediate value first
    data = np.array( generalUtils.sequencePeakAndValleys( data, keepEnds=True ) )
    # search the peak and shift the data
    index = data.argmax( axis=0 )
    n = len( data )
    if index != 0 or index != n - 1:
        for i in range( index ):
            data[ i ] = data[ i + 1 ]
        data = np.roll( data, -index )
        # need to remove the intermediate value again
        data = np.array( generalUtils.sequencePeakAndValleys( data, keepEnds=True ) )

    indices = np.array( range( -1, len( data ) - 1 ) )

    def checkPreviousThree( indices, i ):
        if ( indices[ i ] < 0 ):
            return False
        if ( indices[ indices[ i ] ] < 0 ):
            return False
        return True
    
    rstSeq = [ ]
    # loop from left to right
    i = 2
    while i < len( data ):
        if ( not checkPreviousThree( indices, i ) ): 
            i += 1
            continue
        second = indices[ i ]
        first = indices[ second ]
        left = abs( data[ first ] - data[ second ] )
        right = abs( data[ second ] - data[ i ] )
        if ( left <= right ):
            rstSeq.append( [ data[ first ], data[ second ] ] )
            indices[ i ] = indices[ first ]
            indices[ second ] = -2
            indices[ first ] = -2
        else: 
            i += 1

    if ( not aggregate ): 
        return rstSeq
    
    rstDict = defaultdict( int )
    for leftRight in rstSeq:
        height = round( abs( leftRight[ 1 ] - leftRight[ 0 ] ), globalConfig.atol )
        rstDict[ height ] += 1

    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
