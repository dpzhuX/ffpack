#!/usr/bin/env python3

import numpy as np

def randomWalkUniform( steps, dim=1 ):
    '''
    Random walk function.

    Parameters
    ----------
    steps: scalar
        Number of steps for generating.
    dim: scalar, optional
        Data dimension.
    
    Returns
    -------
    rst: 2d array
        A 2d steps by dim matrix holding the coordinates 
        of the position at each step.
    
    Raises
    ------
    ValueError
        If the steps is less than 1 or the dim is less than 1.

    Examples
    --------
    >>> from ffpack.lsg import randomWalkUniform
    >>> rst = randomWalkUniform( 5 )

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
