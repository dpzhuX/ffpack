#!/usr/bin/env python3

import numpy as np
from collections import defaultdict

def getSequencePeakAndValleys( data, keepEnds=False ):
    '''
    Remove the intermediate value and only get the peak and valley of the data

    The peak and valley refers the data points that are EXACTLY above and below
    the neighbors, not equal. 

    By default, two ends of the sequence data will be removed.

    Args:
        data: 1D input sequence data 
        keepEnds: Keep the ends of the sequence data

    Returns:
        rst: 1D sequence data only contains peak and valley
    
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1 and keepEnds:
        raise ValueError( "Input data length should be at least 2")
    if data.shape[0] <= 2 and not keepEnds:
        raise ValueError( "Input data length should be at least 3")

    rst = []
    prev = None
    for i, cur in enumerate( data ):
        if i == 0 or i == len( data ) - 1:
            if keepEnds:
                rst.append( cur )
        else:
            next = data[ i + 1 ]
            if ( prev < cur and cur > next ) or \
               ( prev > cur and cur < next ):
               rst.append( cur )
        prev = cur
    return rst

def digitizeSequenceToResoultion( data, resolution=1.0 ):
    '''
    Digitize the sequence data to a specific resolution

    Currently, the sequence data are digitized by the round method. By default,
    the resolution is 1.0 so that the sequence data are digitized to the nearest integers.

    Args:
        data: 1D input sequence data
        resoultion: Scalar value of the digitized resolution
    
    Returns:
        rst: 1D digitized data
    
    Notes:
        The default round function will round half to even: 1.5, 2.5 => 2.0:
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )

    rst = []
    for d in data:
        rst.append( np.rint( d / resolution) * resolution )
    return rst

def cycleCountingAccordingToBinSize( data, binSize=1.0 ):
    '''
    Count number of occurrences of each cycle digitized to the nearest bin.

    Args:
        data: 2D array_like
            2D input sequence data [ [ cycle, count ] ] for bin collection 
        binSize: float, optional
            bin size is the difference between each level, 
            for example, binSize=1.0, the levels will be 0.0, 1.0, 2.0, 3.0 ...
    
    Returns:
        rst: 2D array_like
            2D data [ [ level, count ] ] with level starts from 0 to maximum possible value
    
    Notes:
        When a value is in the middle, it will be counted downward
        for example, 0.5 when binSize=1.0, the count will be counted to 0.0 
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
