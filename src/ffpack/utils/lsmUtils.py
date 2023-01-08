#!/usr/bin/env python3

from ffpack.config import globalConfig
import numpy as np


def countingRstToCountingMatrix( countingRst ):
    '''
    Calculate counting matrix from rainflow counting result.

    Parameters
    ----------
    countingRst: 2d array
        Cycle counting result in form of [ [ rangeStart1, rangeEnd1, count1 ], 
        [ rangeStart2, rangeEnd2, count2 ], ... ].
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting results.
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.

    Raises
    ------
    ValueError
        If the data dimension is not 2.
        If the data is not empty and not in dimension of n by 3.

    Examples
    --------
    >>> from ffpack.lsm import countingRstToCountingMatrix
    >>> countingRst = [ [ -2.0, 1.0, 1.0 ], [ 5.0, -1.0, 3.0 ], [ -4.0, 4.0, 0.5 ] ]
    >>> rst, matrixIndexKey = countingRstToCountingMatrix( countingRst )
    '''
    countingRst = np.array( countingRst )
    if len( countingRst.shape ) != 2:
        raise ValueError( "Input data dimension should be 2" )
    if countingRst.shape[ 1 ] != 0 and len( countingRst[ 0 ] ) != 3:
        raise ValueError( "Input data should be either empty or in dimension of n by 3")
    
    matrixIndexKey = np.unique( np.array( countingRst )[ :, 0: 2 ].flatten() )
    matrixIndexKey = [ "{1:,.{0}f}".format( globalConfig.atol, key ) for key in matrixIndexKey ]
    matrixSize = len( matrixIndexKey )
    matrixIndexVal = np.array( [ i for i in range( matrixSize ) ] )
    matrixDict = { k: v for k, v in zip( matrixIndexKey, matrixIndexVal ) }
    if not matrixSize:
        return [ [ ] ], [ ]

    rst = np.zeros( ( matrixSize, matrixSize ) )
    for tuple in countingRst:
        rst[ matrixDict[ "{1:,.{0}f}".format( globalConfig.atol, tuple[ 0 ] ) ],
             matrixDict[ "{1:,.{0}f}".format( globalConfig.atol, tuple[ 1 ] ) ] ] += tuple[ 2 ]

    return rst.tolist(), matrixIndexKey
