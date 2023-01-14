#!/usr/bin/env python3

import numpy as np
from ffpack.utils import generalUtils
from ffpack.config import globalConfig
from collections import defaultdict


def fourPointRainflowCounting( data, aggregate=True ):
    '''
    Four point rainflow counting in [Lee2011]_.

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
        If the data length is less than 4 or the data dimension is not 1.

    Examples
    --------
    >>> from ffpack.lcc import fourPointRainflowCounting
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = fourPointRainflowCounting( data )
    
    References
    ----------
    .. [Lee2011] Lee, Y.L., Barkey, M.E. and Kang, H.T., 2011. Metal fatigue analysis 
       handbook: practical problem-solving techniques for computer-aided engineering. 
       Elsevier.
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] < 4:
        raise ValueError( "Input data length should be at least 4")

    # Remove the intermediate value first
    data = np.array( generalUtils.sequencePeakValleyFilter( data, keepEnds=True ) )
    n = len( data )
    indices = np.array( range( 1, n + 1 ) )
    indices[ n - 1 ] = -1

    def oneRound( indices, data ):
        rst = [ -1, -1, -1, -1 ]
        first = 0
        while ( first != -1 ):
            if indices[ first ] == -1: 
                return rst
            second = indices[ first ]
            if indices[ second ] == -1:
                return rst
            third = indices[ second ]
            if indices[ third ] == -1:
                return rst
            fourth = indices[ third ]
            X = abs( data[ third ] - data[ fourth ] )
            Y = abs( data[ second ] - data[ third ] )
            Z = abs( data[ first ] - data[ second ] )
            if X >= Y and Z >= Y:
                return [ first, second, third, fourth ]
            else: 
                first = second 
        return rst
    
    rstSeq = [ ]
    first = 0
    while first != -1:
        [ first, second, third, fourth ] = oneRound( indices, data )
        if first == -1: 
            break
        rstSeq.append( [ data[ second ], data[ third ], 1 ] )
        indices[ first ] = fourth
        indices[ second ] = -1
        indices[ third ] = -1

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
