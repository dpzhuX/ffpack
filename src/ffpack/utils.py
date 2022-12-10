#!/usr/bin/env python3

import numpy as np

def getPeakAndValley( data, keepEnds=True ):
    '''
    Remove the intermediate value and only get the peak and valley of the data

    The peak and valley refers the data points that are EXACTLY above and below
    the neighbors, not equal. 

    By default, two ends of the sequence data will be preserved.

    Args:
        data: 1D input sequence data 
        keepEnds: Keep the ends of the sequence data

    Returns:
        rst: 1D output sequence data only contains peak and valley
    
    '''
    rst = []
    for i, cur in enumerate( data ):
        if i == 0 or i == len( data ) - 1:
            if keepEnds:
                rst.append( cur )
        else:
            prev = rst[ -1 ]
            next = data[ i + 1 ]
            if ( prev < cur and cur > next ) or \
               ( prev > cur and cur < next ):
               rst.append( cur )
    return rst