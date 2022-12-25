#!/usr/bin/env python3

from ffpack.utils import sequenceDigitization, cycleCountingAggregation
from ffpack.lcc import astmSimpleRangeCounting, astmRainflowCounting
from ffpack.lcc import rychlikRainflowCounting
import numpy as np

def astmSimpleRangeCountingMatrix( data, digitization=True, resolution=0.5 ):

    '''
    Calculate simple range counting matrix

    Parameters
    ----------
    data: 1d array
        Sequence data to calculate range counting matrix.
    digitization: bool
        Sequence digitization before the cycle counting.
        Otherwise, the counting resutls are aggregated after cycle counting.
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting resutls.
    
    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2.

    Notes
    -----
    The default round function will round half to even: 1.5, 2.5 => 2.0:

    Examples
    --------
    >>> from ffpack.lsm import astmSimpleRangeCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmSimpleRangeCountingMatrix( data )
    '''

    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    if digitization:
        data = sequenceDigitization( data, resolution )
    
    countingRst = astmSimpleRangeCounting( data, aggregate=False )

    if not digitization:
        countingRst = cycleCountingAggregation( countingRst, binSize=resolution )
    
    matrixIndexKey = np.unique( np.array( countingRst ).flatten() )
    matrixSize = len( matrixIndexKey )
    matrixIndexVal = np.array( [ i for i in range( matrixSize ) ] )
    matrixDict = { k: v for k, v in zip( matrixIndexKey, matrixIndexVal ) }
    rst = np.zeros( ( matrixSize, matrixSize ) )

    for pair in countingRst:
        rst[ matrixDict[ pair[ 0 ] ], matrixDict[ pair[ 1 ] ] ] = 1
    return rst.tolist()


def astmRainflowCountingMatrix():
    pass


def rychlikRainflowCountingmatrix():
    pass