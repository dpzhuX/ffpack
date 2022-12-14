#!/usr/bin/env python3

import numpy as np
from ffpack.utils import generalUtils 
from collections import defaultdict 

def rychlikRainflowCycleCounting( data, aggregate=True ):
    '''
    Rychilk rainflow counting (topLevel-up cycle TUC) in 
    "A new definition of the rainflow cycle counting method" by Rychilk on IJF.

    Parameters
    ----------
    data: 1darray 
        Load sequence data for counting.
    aggragate: bool, optional
        if aggregate set to False, the original range H(t) sequence will be returned.
    
    Returns
    -------
    rst: 2darray
        Sorted counting restults.
    
    Notes
    -----
    If aggregate is False, the original 1d counting resutls will be returned.
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


    peakVallays = generalUtils.getSequencePeakAndValleys( data, keepEnds=True )
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
