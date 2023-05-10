#!/usr/bin/env python3

import numpy as np


def randomWalkUniform( numSteps, dim=1, randomSeed=None ):
    '''
    Generate load sequence by a random walk.

    Parameters
    ----------
    numSteps: integer 
        Number of steps for generating.
    dim: scalar, optional
        Data dimension.
    randomSeed: integer, optional
        Random seed. If randomSeed is none or is not an integer, the random seed in 
        global config will be used. 
    
    Returns
    -------
    rst: 2d array
        A 2d (numSteps by dim) matrix holding the coordinates 
        of the position at each step.
    
    Raises
    ------
    ValueError
        If the numSteps is less than 1 or the dim is less than 1.

    Examples
    --------
    >>> from ffpack.lsg import randomWalkUniform
    >>> rst = randomWalkUniform( 5 )

    '''
    # Edge case check
    if not isinstance( numSteps, int ) or not isinstance( dim, int ):
        raise ValueError( "numSteps should be int" )
    if numSteps < 1:
        raise ValueError( "numSteps should be at least 1" )
    if dim < 1:
        raise ValueError( "dim should be at least 1" )

    if isinstance(randomSeed, (int, type(None))):
        np.random.seed( randomSeed )
    
    rst = [ [ 0 ] * dim ]
    for i in range( numSteps ):
        randomInt = np.random.randint( 2 * dim )
        randomDim = randomInt % dim
        randomDir = 1 if randomInt >= dim else -1
        lastCoords = np.copy( rst[ -1 ] )
        lastCoords[ randomDim ] += randomDir
        rst.append( lastCoords.tolist() )
    return rst
