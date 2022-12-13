#!/usr/bin/env python3

import numpy as np
from ffpack import utils
from collections import defaultdict, deque

def rychlikRainflowCycleCounting( data, aggregate=True ):
    '''
    Implement the rainflow counting method based on Definition 1 (topLevel-up cycle TUC)
    in "A new definition of the rainflow cycle counting method" by Rychilk on IJF
    Args:
        data: array_like
            1D input sequence data for counting
        aggragate: bool, optional
            if aggregate the range H(t) sequence nor not

    Returns:
        rst: 1D range H(t) sequence if aggregate is false
             2D sorted output data if aggregate is true
    '''

    def getMinLeft( data, i ):
        if ( i == 1 ): 
            return min( data[ 0 ], data[ 1 ] )
        
        left = data[ i - 1 ]
        j = i - 2
        while ( j >= 0 and data[ j ] < data[ i ] ): 
            left = min( left, data[ j ] )
            j -= 1

        return left

    def getMinRight( data, i ):
        if ( i == len( data ) - 2 ): 
            return min( data[ len( data ) - 1 ], data[ len( data ) - 2 ] )
        
        right = data[ i + 1 ]
        j = i + 2
        while ( j < len( data ) and data[ j ] < data[ i ] ):
            right = min( right, data[ j ] )
            j += 1

        return right 


    peakVallays = utils.getSequencePeakAndValleys( data, keepEnds=True )
    rstSeq = [ ]
    for i in range( 1, len( data ) - 1 ):
        if ( data[ i ] > data[ i - 1 ] and data[ i ] > data[ i + 1 ] ):
            # TODO: set a global variable for round digit
            height = data[ i ] - max( getMinLeft( data, i ), getMinRight( data, i ))
            rstSeq.append( round( height, 7 ) )
    
    if ( not aggregate ): 
        return rstSeq

    rstDict = defaultdict( int )
    for i, cur in enumerate( rstSeq ):
        rstDict[ cur ] += 1

    if len( rstDict ) == 0:
        return [ [ ] ] 
    rst = np.array( [ [ key, val ] for key, val in rstDict.items() ] )
    rst = rst[ rst[ :, 0 ].argsort() ]
    return rst.tolist()
