#!/usr/bin/env python3

'''
Rychlik proposed a toplevel-up cycle counting method and proved that the proposed 
method is equivalent to the classical rainflow counting method. Compared to the
classical rainflow counting method, the proposed method keeps the original 
sequence information which is quite useful if the sequence information is
required for further analysis.

Reference: Rychlik, I., 1987. A new definition of the rainflow cycle counting method. 
International journal of fatigue, 9(2), pp.119-121.
'''

import numpy as np
from ffpack.utils import sequenceFilter 
from ffpack.config import globalConfig
from collections import defaultdict 


def rychlikRainflowCounting( data, aggregate=True ):
    '''
    Rychilk rainflow counting (toplevel-up cycle TUC)

    Parameters
    ----------
    data: 1d array 
        Load sequence data for counting.
    aggragate: bool, optional
        If aggregate is set to False, the original sequence for internal counting,
        e.g., [ [ rangeStart1, rangeEnd1, count1 ], [ rangeStart2, rangeEnd2, count2 ], ... ], 
        will be returned.
    
    Returns
    -------
    rst: 2d array
        Sorted counting results.

    Raises
    ------
    ValueError
        If the data dimension is not 1
        If the data length is less than 2

    Examples
    --------
    >>> from ffpack.lcc import rychlikRainflowCycleCounting
    >>> data = [ -0.8, 1.3, 0.7, 3.4, 0.7, 2.5, -1.4, -0.5, -2.3, 
    >>>          -2.2, -2.6, -2.4, -3.3, 1.5, 0.6, 3.4, -0.5 ]
    >>> rst = rychlikRainflowCycleCounting( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] < 2:
        raise ValueError( "Input data length should be at least 2" )
    
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

    # we need to use this util function since it keeps one peak 
    # if there are two or more points together with the same peak value
    data = sequenceFilter.sequencePeakValleyFilter( data, keepEnds=True )
    rstSeq = [ ]
    for i in range( 1, len( data ) - 1 ):
        if ( data[ i ] > data[ i - 1 ] and data[ i ] > data[ i + 1 ] ):
            higher = max( getMinLeft( data, i ), getMinRight( data, i ) )
            rstSeq.append( [ higher, data[ i ], 1 ] )
    
    if ( not aggregate ): 
        return rstSeq

    rstDict = defaultdict( int )
    for lowHigh in rstSeq:
        height = round( lowHigh[ 1 ] - lowHigh[ 0 ], globalConfig.atol )
        rstDict[ height ] += 1

    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
