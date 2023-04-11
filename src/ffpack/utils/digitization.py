#!/usr/bin/env python3

import numpy as np


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
    data = np.array( data, dtype=float )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )

    rst = [ ]
    for d in data:
        rst.append( np.rint( d / resolution) * resolution )
    return rst
