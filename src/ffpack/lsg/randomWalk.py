#!/usr/bin/env python3

import numpy as np

def randomWalk( steps, dim=1 ):
    '''
    Implement the random walk function.

    Args:
        steps: scalar value denoting the number of steps
        dim: scalar value denoting the data dimension

    Returns:
        rst: 2D steps by dim matrix holding the coordinates 
             of the position at each step.
    '''
    # Edge case check
    if not isinstance( steps, int ) or not isinstance( dim, int ):
        raise ValueError( "Input data type should be int" )
    if steps < 1:
        raise ValueError( "steps should be at least 1" )
    if dim < 1:
        raise ValueError( "dim should be at least 1" )

    rst = [ [ 0 ] * dim ]
    for i in range( steps ):
        randomInt = np.random.randint( 2 * dim )
        randomDim = randomInt % dim
        randomDir = 1 if randomInt >= dim else -1
        lastCoords = np.copy( rst[ -1 ] )
        lastCoords[ randomDim ] += randomDir
        rst.append( lastCoords.tolist() )
    return rst
