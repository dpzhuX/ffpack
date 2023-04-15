#!/usr/bin/env python3

import numpy as np
from collections import defaultdict


def cycleCountingAggregation( data, binSize=1.0 ):
    '''
    Count the number of occurrences of each cycle digitized to the nearest bin.

    Parameters
    ----------
    data: 2d array
        Input cycle counting data [ [ value, count ], ... ] for bin collection 

    binSize: scalar, optional
        bin size is the difference between each level, 
        for example, binSize=1.0, the levels will be 0.0, 1.0, 2.0, 3.0 ...

    Returns
    -------
    rst: 2d array
        Aggregated [ [ aggregatedValue, count ] ] by the binSize


    Raises
    ------
    ValueError
        If the data dimension is not 2.
        If the data is empty

    Notes
    -----
    When a value is in the middle, it will be counted downward
    for example, 0.5 when binSize=1.0, the count will be counted to 0.0 

    Examples
    --------
    >>> from ffpack.utils import cycleCountingAggregation
    >>> data = [ [ 1.7, 2.0 ], [ 2.2, 2.0 ] ]
    >>> rst = cycleCountingAggregation( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 2:
        raise ValueError( "Input data dimension should be 2" )
    if data.shape[1] != 2:
        raise ValueError( "Input data should be [ value, count ] pairs")

    def getBinKey( value ):
        key = binSize * int( value / binSize )
        if ( value - key > key + binSize - value):
            key += binSize
        return key

    rstDict = defaultdict( int )
    for valueCount in data:
        key = getBinKey( valueCount[ 0 ] )
        rstDict[ key ] += valueCount[ 1 ]
    
    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
