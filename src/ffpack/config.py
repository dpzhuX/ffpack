#!/usr/bin/env python3

import numpy as np

class GlobalConfig:
    '''
    Global config for FFPACK
    '''
    def __init__( self ):
        '''
        Initialize a global config instance with default values.

        Attributes
        ----------
        seed: scalar
            Seed for random number generator. 
            Default value is None.
        atol: scalar
            Absolute tolerance in digits.
            Default value is 8.
        rtol: scalar
            Relative tolerance in digits.
            Default value is 5.

        Examples
        --------
        >>> from ffpack.config import globalConfig
        >>> globalConfig.atol = 7
        '''
        # Seed for random number generator
        self.seed = None
        # Absolute tolerance in digits
        self.atol = 8
        # Relative tolerance in digits
        self.rtol = 5
    
    def setSeed( self, seed ):
        '''
        Set seed for random number generator

        Parameters
        ----------
        seed: scalar
            Input seed for random number generator
        
        Notes
        -----
        Set seed to None can clean the seed
        
        Examples
        --------
        >>> from ffpack.config import globalConfig
        >>> globalConfig.setSeed( 0 )
        '''
        self.seed = seed
        np.random.seed( self.seed )


globalConfig = GlobalConfig()
