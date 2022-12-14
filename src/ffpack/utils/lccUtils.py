#!/usr/bin/env python3

import numpy as np
from collections import defaultdict

def cycleCountingAccordingToBinSize( data, binSize=1.0 ):
    '''
    Count number of occurrences of each cycle digitized to the nearest bin.

    Parameters
    ----------
    data: 2d array
        Input load cycle counting data [ [ range, count ], ... ] for bin collection 

    binSize: scalar, optional
        bin size is the difference between each level, 
        for example, binSize=1.0, the levels will be 0.0, 1.0, 2.0, 3.0 ...

    Returns
    -------
    rst: 2d array
        Aggregated [ [ range, count ] ] with range starts from 0 to maximum 
        possible value by the binSize


    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2 with keedEnds == False
        If the data length is less than 3 with keedEnds == True

    Notes
    -----
    When a value is in the middle, it will be counted downward
    for example, 0.5 when binSize=1.0, the count will be counted to 0.0 

    Examples
    --------
    >>> from ffpack.utils import cycleCountingAccordingToBinSize
    >>> data = [ [ 1.7, 2.0 ], [ 2.2, 2.0 ] ]
    >>> rst = cycleCountingAccordingToBinSize( data )
    '''
    def getBinKey( value ):
        key = binSize * int( value / binSize )
        if ( value - key > key + binSize - value):
            key += binSize
        return key

    rstDict = defaultdict( int )
    max_value = np.max( np.array( data )[ :, 0 ] )
    numBins = int( getBinKey( max_value ) / binSize ) + 1
    for i in range( numBins ):
        rstDict[ i * binSize ] = 0

    for valueCount in data:
        key = getBinKey( valueCount[ 0 ] )
        rstDict[ key ] += valueCount[ 1 ]
    
    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
