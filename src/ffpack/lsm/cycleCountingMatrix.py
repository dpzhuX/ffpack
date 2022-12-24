#!/usr/bin/env python3

from ffpack.utils import sequenceDigitization, cycleCountingAggregation
from ffpack.lcc import astmSimpleRangeCounting, astmRainflowCounting
from ffpack.lcc import rychlikRainflowCounting
import numpy as np

def astmSimpleRangeCountingMatrix( data, method, digitization=True, resolution=0.5 ):

    if digitization:
        data = sequenceDigitization( data, resolution )
    
    countingRst = astmSimpleRangeCounting( data, aggregate=False )

    if not digitization:
        countingRst = cycleCountingAggregation( countingRst, binSize=resolution )
    
    matrixIndex = np.unique( np.array( countingRst ).flatten() )
    matrixSize = len( matrixIndex )
    rst = np.zeros( ( matrixSize, matrixSize ) )
    for pair in countingRst:
        rst[ pair[ 0 ], pair[ 1 ] ] = 1
    return rst


def astmRainflowCountingMatrix():
    pass


def rychlikRainflowCountingmatrix():
    pass