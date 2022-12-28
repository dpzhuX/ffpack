#!/usr/bin/env python3

import numpy as np

class MetropolisHastingsSampler:
    def __init__( self, initialVal=None, targetPdf=None, proposalCSampler=None ):
        self.cur = initialVal
        self.nxt = initialVal
        self.targetPdf = targetPdf
        self.proposalCSampler = proposalCSampler
    
    def getAcceptanceRatio( self, candi ):
        fcur = self.targetPdf( self.cur )
        fcandi = self.targetPdf( candi )
        return fcandi / fcur
    
    def getCandidate( self ):
        return self.proposalCSampler( self.cur )

    def sample( self ):
        candi = self.getCandidate()
        acceptanceRatio = self.getAcceptanceRatio( candi )
        u = np.random.uniform()
        if u <= acceptanceRatio:
            self.nxt = candi
        else:
            self.nxt = self.cur
        self.cur = self.nxt
        return self.cur
