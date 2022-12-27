#!/usr/bin/env python3

import numpy as np

class MetropolisHastingsSampler:
    def __init__( self, initialVal=None, targetPdf=None, proposalCpdf=None ):
        self.cur = initialVal
        self.nxt = initialVal
        self.targetPdf = targetPdf
        self.proposalCpdf = proposalCpdf
    
    def getAcceptanceRatio( self, candi ):
        fcur = self.targetPdf( self.cur )
        fcandi = self.targetPdf( candi )
        return fcandi / fcur
    
    def getCandidate( self ):
        return self.proposalCpdf( self.cur )

    def sample( self ):
        candi = self.getCandidate()
        acceptanceRatio = self.getAcceptanceRatio()
        u = np.random.uniform()
        if u <= acceptanceRatio:
            self.nxt = candi
        else:
            self.nxt = self.cur
        self.cur = self.nxt
        return self.cur
