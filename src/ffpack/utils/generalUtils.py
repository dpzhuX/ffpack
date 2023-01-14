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
        If the data length is less than 2 with keedEnds == False
        If the data length is less than 3 with keedEnds == True

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
        raise ValueError( "Input data length should be at least 2")
    if data.shape[0] <= 2 and not keepEnds:
        raise ValueError( "Input data length should be at least 3")

    rst = []
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
