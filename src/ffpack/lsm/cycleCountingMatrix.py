#!/usr/bin/env python3

from ffpack.utils import generalUtils 
from ffpack.lcc import astmCounting
from ffpack.lcc import rychlikCounting
from ffpack.config import globalConfig
import numpy as np

def astmSimpleRangeCountingMatrix( data, resolution=0.5 ):
    '''
    Calculate ASTM simple range counting matrix.

    Parameters
    ----------
    data: 1d array
        Sequence data to calculate range counting matrix.
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting results.
    
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

    data = generalUtils.sequenceDigitization( data, resolution )
    
    countingRst = astmCounting.astmSimpleRangeCounting( data, aggregate=False )

    matrixIndexKey = np.unique( np.array( countingRst ).flatten() )
    matrixIndexKey = [ "{1:,.{0}f}".format( globalConfig.atol, key ) for key in matrixIndexKey ]
    matrixSize = len( matrixIndexKey )
    matrixIndexVal = np.array( [ i for i in range( matrixSize ) ] )
    matrixDict = { k: v for k, v in zip( matrixIndexKey, matrixIndexVal ) }
    if not matrixSize:
        return [ [ ] ], [ ]

    rst = np.zeros( ( matrixSize, matrixSize ) )

    for pair in countingRst:
        rst[ matrixDict[ "{1:,.{0}f}".format( globalConfig.atol, pair[ 0 ] ) ], 
             matrixDict[ "{1:,.{0}f}".format( globalConfig.atol, pair[ 1 ] ) ] ] += 0.5
    return rst.tolist(), matrixIndexKey


def astmRainflowCountingMatrix( data, resolution=0.5 ):
    '''
    Calculate ASTM rainflow counting matrix.

    Parameters
    ----------
    data: 1d array
        Sequence data to calculate rainflow counting matrix.
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting results.
    
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
    >>> from ffpack.lsm import astmRainflowCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmRainflowCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    
    countingRst = astmCounting.astmRainflowCounting( data, aggregate=False )

    matrixIndexKey = np.unique( np.array( countingRst )[ :, 0:-1 ].flatten() )
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


def rychlikRainflowCountingMatrix( data, resolution=0.5 ):
    '''
    Calculate Rychlik rainflow counting matrix.

    Parameters
    ----------
    data: 1d array
        Sequence data to calculate rainflow counting matrix.
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting results.
    
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
    >>> from ffpack.lsm import astmRainflowCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst = astmRainflowCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    
    countingRst = rychlikCounting.rychlikRainflowCounting( data, aggregate=False )

    matrixIndexKey = np.unique( np.array( countingRst ).flatten() )
    matrixIndexKey = [ "{1:,.{0}f}".format( globalConfig.atol, key ) for key in matrixIndexKey ]
    matrixSize = len( matrixIndexKey )
    matrixIndexVal = np.array( [ i for i in range( matrixSize ) ] )
    matrixDict = { k: v for k, v in zip( matrixIndexKey, matrixIndexVal ) }
    if not matrixSize:
        return [ [ ] ], [ ]

    rst = np.zeros( ( matrixSize, matrixSize ) )

    for pair in countingRst:
        rst[ matrixDict[ "{1:,.{0}f}".format( globalConfig.atol, pair[ 0 ] ) ], 
             matrixDict[ "{1:,.{0}f}".format( globalConfig.atol, pair[ 1 ] ) ] ] += 1.0
    return rst.tolist(), matrixIndexKey
