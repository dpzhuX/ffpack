#!/usr/bin/env python3

from ffpack.lcc import astmCounting
from ffpack.lcc import rychlikCounting
from ffpack.lcc import johannessonCounting
from ffpack.utils import generalUtils 
from ffpack.utils import lsmUtils 
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
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.

    Raises
    ------
    ValueError
        If the data dimension is not 1.
        If the data length is less than 2.

    Notes
    -----
    The default round function will round half to even: 1.5, 2.5 => 2.0.

    Examples
    --------
    >>> from ffpack.lsm import astmSimpleRangeCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst, matrixIndexKey = astmSimpleRangeCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    countingRst = astmCounting.astmSimpleRangeCounting( data, aggregate=False )
    return lsmUtils.countingRstToCountingMatrix( countingRst )



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
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.
       
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
    >>> rst, matrixIndexKey = astmRainflowCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    countingRst = astmCounting.astmRainflowCounting( data, aggregate=False )
    return lsmUtils.countingRstToCountingMatrix( countingRst )



def astmRangePairCountingMatrix( data, resolution=0.5 ):
    '''
    Calculate ASTM range pair counting matrix.

    Parameters
    ----------
    data: 1d array
        Sequence data to calculate range pair counting matrix.
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting results.
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.
    
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
    >>> from ffpack.lsm import astmRangePairCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst, matrixIndexKey = astmRangePairCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    countingRst = astmCounting.astmRangePairCounting( data, aggregate=False )
    return lsmUtils.countingRstToCountingMatrix( countingRst )



def astmRainflowRepeatHistoryCountingMatrix( data, resolution=0.5 ):
    '''
    Calculate ASTM simplified rainflow counting matrix for repeating histories.

    Parameters
    ----------
    data: 1d array
        Sequence data to calculate simplified rainflow counting matrix 
        for repeating histories.
    resolution: bool, optional
        The desired resolution to round the data points.
    
    Returns
    -------
    rst: 2d array
        A matrix contains the counting results.
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.
    
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
    >>> from ffpack.lsm import astmRainflowRepeatHistoryCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst, matrixIndexKey = astmRainflowRepeatHistoryCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    countingRst = astmCounting.astmRainflowRepeatHistoryCounting( data, aggregate=False )
    return lsmUtils.countingRstToCountingMatrix( countingRst )



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
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.
    
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
    >>> from ffpack.lsm import rychlikRainflowCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst, matrixIndexKey = rychlikRainflowCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    countingRst = rychlikCounting.rychlikRainflowCounting( data, aggregate=False )
    return lsmUtils.countingRstToCountingMatrix( countingRst )



def johannessonMinMaxCountingMatrix( data, resolution=0.5 ):
    '''
    Calculate Johannesson minMax cycle counting matrix.

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
    matrixIndexKey: 1d array
        A sorted array contains the index keys for the counting matrix.
    
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
    >>> from ffpack.lsm import johannessonMinMaxCountingMatrix
    >>> data = [ -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0 ]
    >>> rst, matrixIndexKey = johannessonMinMaxCountingMatrix( data )
    '''
    data = np.array( data )
    if len( data.shape ) != 1:
        raise ValueError( "Input data dimension should be 1" )
    if data.shape[0] <= 1:
        raise ValueError( "Input data length should be at least 2")

    data = generalUtils.sequenceDigitization( data, resolution )
    countingRst = johannessonCounting.johannessonMinMaxCounting( data, aggregate=False )
    return lsmUtils.countingRstToCountingMatrix( countingRst )
