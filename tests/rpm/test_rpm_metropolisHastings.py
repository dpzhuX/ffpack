#!/usr/bin/env python3

from ffpack import rpm
import numpy as np
import pytest
from unittest.mock import patch


###############################################################################
# Test MetropolisHastingsSampler
###############################################################################
def test_MetropolisHastingsSampler_initialValNone_valueError():
    initialVal = None

    def targetPdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    def proposalCSampler( x ):
        return np.random.normal( x, 1 )
    
    with pytest.raises( ValueError ):
        _ = rpm.MetropolisHastingsSampler( initialVal=initialVal, 
                                           targetPdf=targetPdf, 
                                           proposalCSampler=proposalCSampler )
    

def test_MetropolisHastingsSampler_targetPdfNone_valueError():
    initialVal = 1

    targetPdf = None

    def proposalCSampler( x ):
        return np.random.normal( x, 1 )
    
    with pytest.raises( ValueError ):
        _ = rpm.MetropolisHastingsSampler( initialVal=initialVal, 
                                           targetPdf=targetPdf, 
                                           proposalCSampler=proposalCSampler )


def test_MetropolisHastingsSampler_proposalCSamplerNone_valueError():
    initialVal = 1

    def targetPdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    proposalCSampler = None
    
    with pytest.raises( ValueError ):
        _ = rpm.MetropolisHastingsSampler( initialVal=initialVal, 
                                           targetPdf=targetPdf, 
                                           proposalCSampler=proposalCSampler )
    

@patch( "numpy.random.normal" )
def test_MetropolisHastingsSampler_targetPdfNegativeReturn_valueError( mock_normal ):
    mock_normal.return_value = 1
    initialVal = 1

    def targetPdf( x ):
        return -1
    
    def proposalCSampler( x ):
        return np.random.normal( x, 1 )
    
    with pytest.raises( ValueError ):
        mhSampler = rpm.MetropolisHastingsSampler( initialVal=initialVal, 
                                                   targetPdf=targetPdf, 
                                                   proposalCSampler=proposalCSampler )
        mhSampler.getSample()
    

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



###############################################################################
# Test AuModifiedMHSampler
###############################################################################
def test_AuModifiedMHSampler_initialValError_valueError():
    initialVal = None

    def tpdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    targetPdf = [ tpdf, tpdf ]
    
    def pcs( x ):
        return np.random.normal( x, 1 )
    
    proposalCSampler = [ pcs, pcs ]
    
    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )
    
    initialVal = 1.0
    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )


def test_AuModifiedMHSampler_targetPdfError_valueError():
    initialVal = [ 1.0, 1.0 ]

    targetPdf = None

    def pcs( x ):
        return np.random.normal( x, 1 )
    
    proposalCSampler = [ pcs, pcs ]
    
    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )

    def tpdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    targetPdf = tpdf

    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )


def test_AuModifiedMHSampler_proposalCSamplerError_valueError():
    initialVal = [ 1.0, 1.0 ]

    def tpdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    targetPdf = [ tpdf, tpdf ]
    
    proposalCSampler = None
    
    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )
    
    def pcs( x ):
        return np.random.normal( x, 1 )
    
    proposalCSampler = pcs

    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )


def test_AuModifiedMHSampler_dimNotEqual_valueError():
    def tpdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    def pcs( x ):
        return np.random.normal( x, 1 )
    
    initialVal = [ 1.0, 1.0 ]
    targetPdf = [ tpdf, tpdf ]
    proposalCSampler = [ pcs, pcs, pcs ]
    
    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )

    initialVal = [ 1.0, 1.0 ]
    targetPdf = [ tpdf, tpdf, tpdf ]
    proposalCSampler = [ pcs, pcs ]
    
    with pytest.raises( ValueError ):
        _ = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                     targetPdf=targetPdf, 
                                     proposalCSampler=proposalCSampler )


@patch( "numpy.random.normal" )
def test_AuModifiedMHSampler_targetPdfNegativeReturn_valueError( mock_normal ):
    mock_normal.return_value = 1
    initialVal = [ 1.0, 1.0 ]

    def tpdf( x ):
        return -1
    
    targetPdf = [ tpdf, tpdf ]
    
    def pcs( x ):
        return np.random.normal( x, 1 )
    
    proposalCSampler = [ pcs, pcs ]
    
    with pytest.raises( ValueError ):
        auMMHSampler = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                                targetPdf=targetPdf, 
                                                proposalCSampler=proposalCSampler )
        auMMHSampler.getSample()


@patch( "numpy.random.normal" )
@patch( "numpy.random.uniform" )
@pytest.mark.parametrize( "pseudoUniformVal, pseudoNormalVal", 
                          [ ( 0.2, 0.5 ), ( 0.5, 0.7 ), ( 0.3, 1.5 ), ( 0.4, -1.5 ),
                            ( 0.6, -2.0 ), ( 0.1, 0.0 ), ( 0.8, 1.6 ), ( 0.9, -3.5 ),
                            ( 0.4, -0.6 ), ( 0.2, 6.5 ), ( 0.7, -6.5 ) ] )
def test_AuModifiedMHSampler_normalUseCase_diffByOne( 
        mock_uniform, mock_normal, pseudoUniformVal, pseudoNormalVal ):
    mock_uniform.return_value = pseudoUniformVal
    mock_normal.return_value = pseudoNormalVal

    initialVal = [ 1.0, 1.0 ]

    def tpdf( x ):
        if x < 0:
            return 0
        else:
            return np.exp( -x )
    
    targetPdf = [ tpdf, tpdf ]
    
    def pcs( x ):
        return np.random.normal( x, 1 )
    
    proposalCSampler = [ pcs, pcs ]
    
    auMMHSampler = rpm.AuModifiedMHSampler( initialVal=initialVal, 
                                            targetPdf=targetPdf, 
                                            proposalCSampler=proposalCSampler )

    candi0 = auMMHSampler.getCandidate( 0 )
    acceptanceRatio0 = auMMHSampler.getAcceptanceRatio( candi0, 0 )
    candi1 = auMMHSampler.getCandidate( 1 )
    acceptanceRatio1 = auMMHSampler.getAcceptanceRatio( candi1, 0 )
    np.testing.assert_allclose( candi0, pseudoNormalVal )
    np.testing.assert_allclose( candi1, pseudoNormalVal )
    np.testing.assert_allclose( acceptanceRatio0, 
                                targetPdf[ 0 ]( candi0 ) / targetPdf[ 0 ]( initialVal[ 0 ] ) )
    np.testing.assert_allclose( acceptanceRatio1, 
                                targetPdf[ 1 ]( candi1 ) / targetPdf[ 1 ]( initialVal[ 1 ] ) )

    # Once get the sample, internal state will be changed.
    sampleRst = auMMHSampler.getSample()
    np.testing.assert_allclose( 
        sampleRst[ 0 ], candi0 if pseudoUniformVal <= acceptanceRatio0 else initialVal[ 0 ] )
    np.testing.assert_allclose( 
        sampleRst[ 1 ], candi0 if pseudoUniformVal <= acceptanceRatio1 else initialVal[ 1 ] )
