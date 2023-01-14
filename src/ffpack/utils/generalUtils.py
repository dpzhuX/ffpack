#!/usr/bin/env python3

import numpy as np


def sequencePeakValleyFilter( data, keepEnds=False ):
    '''
    Remove the intermediate value and only get the peaks and valleys of the data

    The peak and valley refer the data points that are EXACTLY above and below
    the neighbors, not equal. 

    Parameters
    ----------
    data: 1darray
        Sequence data to get peaks and valleys.
    keepEnds: bool, optional
        If two ends of the original data should be preserved.
    
    Returns
    -------
    rst: 1darray
        A list contains the peaks and valleys of the data.
    
    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2 with keedEnds == False.
        If the data length is less than 3 with keedEnds == True.

    Examples
    --------
    >>> from ffpack.utils import sequencePeakValleyFilter
    >>> data = [ -0.5, 1.0, -2.0, 3.0, -1.0, 4.5, -2.5, 3.5, -1.5, 1.0 ]
    >>> rst = sequencePeakValleyFilter( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1 and keepEnds:
        raise ValueError( "Input data length should be at least 2" )
    if data.shape[0] <= 2 and not keepEnds:
        raise ValueError( "Input data length should be at least 3" )

    rst = [ ]
    prev = data[ 0 ]
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



def sequenceHysteresisFilter( data, gateSize ):
    '''
    Filter data within the gateSize.

    Any cycle that has an amplitude smaller than the gate is removed from the data. 
    This is done by scan the data, i.e., point i, to check if the next points, 
    i.e., i + 1, i + 2, ... are within the gate from point i. If the next point 
    not within the gate, find and keep only the peak or valley. 

    Parameters
    ----------
    data: 1darray
        Sequence data to get peaks and valleys.
    gateSize: scalar
        Gate size to filter the data. 
    
    Returns
    -------
    rst: 1darray
        A list contains the filtered data.
    
    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2.
        If gateSize is not a scalar or not positive. 

    Examples
    --------
    >>> from ffpack.utils import sequenceHysteresisFilter
    >>> data = [ 2, 5, 3, 6, 2, 4, 1, 6, 1, 3, 1, 5, 3, 6, 3, 6, 4, 5, 2 ]
    >>> rst = sequenceHysteresisFilter( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] < 2:
        raise ValueError( "Input data length should be at least 2" )
    if not isinstance( gateSize, int ) and not isinstance( gateSize, float ):
        raise ValueError( "gateSize must be a scalar" )
    if gateSize <= 0:
        raise ValueError( "gateSize should be greater than zero" )

    n = len( data )
    keep = [ 1 ] * n

    def findPeak( i ):
        # find the index after i that is the peak
        # return last index if no peak is found
        while ( i < n - 1 ):
            if ( data[ i ] > data[ i + 1 ] ):
                return i
            else:
                keep[ i ] = -1
                i += 1
        return n - 1

    def findValley( i ):
        # find the index after i that is the peak
        # return last index if no peak is found
        while ( i < n - 1 ):
            if ( data[ i ] < data[ i + 1 ] ):
                return i
            else:
                keep[ i ] = -1
                i += 1
        return n - 1

    i = 0
    while ( i < n - 1 ):
        cur = data[ i ]
        next = data[ i + 1 ]
        if ( next == cur ):
            keep[ i + 1 ] = -1
            i += 2
            continue
        # search for next point not within gate
        j = i + 1
        if ( next > cur ):
            while ( j < n - 1 ):
                if ( data[ j ] < cur ):
                    break
                elif ( cur + gateSize > data[ j ] ):
                    keep[ j ] = -1
                    j += 1
                else:
                    j = findPeak( j )
                    break
            i = j
        else:
            while ( j < n - 1 ):
                if ( data[ j ] > cur ):
                    break
                elif ( cur - gateSize < data[ j ] ):
                    keep[ j ] = -1
                    j += 1
                else:
                    j = findValley( j )
                    break
            i = j
    
    rst = [ ]
    for i in range( n ):
        if ( keep[ i ] > 0 ):
            rst.append( data[ i ] )
    
    return rst



def sequenceDigitization( data, resolution=1.0 ):
    '''
    Digitize the sequence data to a specific resolution

    The sequence data are digitized by the round method. 

    Parameters
    ----------
    data: 1d array
        Sequence data to digitize.
    
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 1d array
        A list contains the digitized data.
    
    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2 with keedEnds == False
        If the data length is less than 3 with keedEnds == True

    Notes
    -----
    The default round function will round half to even: 1.5, 2.5 => 2.0:

    Examples
    --------
    >>> from ffpack.utils import sequenceDigitization 
    >>> data = [ -1.0, 2.3, 1.8, 0.6, -0.4, 0.8, -1.6, -2.5, 3.4, 0.3, 0.1 ]
    >>> rst = sequenceDigitization( data )
    '''
    # Egde cases
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )

    rst = []
    for d in data:
        rst.append( np.rint( d / resolution) * resolution )
    return rst
