#!/usr/bin/env python3

from ffpack import rpm
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test MetropolisHastingsSampler
###############################################################################
@patch( "numpy.random.normal" )
@patch( "numpy.random.uniform" )
@pytest.mark.parametrize( "pseudoUniformVal, pseudoNormalVal", 
                          [ ( 0.2, 0.5 ), ( 0.5, 0.7 ), ( 0.3, 1.5 ), ( 0.4, -1.5 ),
                            ( 0.6, -2.0 ), ( 0.1, 0.0 ), ( 0.8, 1.6 ), ( 0.9, -3.5 ),
                            ( 0.4, -0.6 ), ( 0.2, 6.5 ), ( 0.7, -6.5 ) ] )
def test_MetropolisHastingsSampler_normalUseCase_diffByOne( 
        mock_uniform, mock_normal, pseudoUniformVal, pseudoNormalVal ):
    mock_uniform.return_value = pseudoUniformVal
    mock_normal.return_value = pseudoNormalVal

    initialVal = 1

    def targetPdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    def proposalCSampler( x ):
        return np.random.normal( x, 1 )
    
    mhSampler = rpm.MetropolisHastingsSampler( initialVal=initialVal, 
                                               targetPdf=targetPdf, 
                                               proposalCSampler=proposalCSampler )
    
    candi = mhSampler.getCandidate()
    acceptanceRatio = mhSampler.getAcceptanceRatio( candi )
    sampleRst = mhSampler.getSample()
    np.testing.assert_allclose( candi, pseudoNormalVal )
    np.testing.assert_allclose( acceptanceRatio, 
                                targetPdf( candi ) / targetPdf( initialVal ) )
    np.testing.assert_allclose( 
        sampleRst, candi if pseudoUniformVal <= acceptanceRatio else initialVal )
