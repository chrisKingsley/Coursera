#!/usr/bin/env python

import os, re, sys

def readTransitionFile(filePath, emissionProbs=False):
    infile = open(filePath, 'r')
    
    if emissionProbs:
        emittedSeq = infile.readline().rstrip()
        infile.readline();infile.readline();infile.readline()
        
    hiddenSeq = infile.readline().rstrip()
    infile.readline();infile.readline();infile.readline()
    
    transMat = dict()
    states1 = infile.readline().rstrip().split()
    for line in infile:
        tokens = line.rstrip().split()
        state2 = tokens[0]
        for i in range(1, len(tokens)):
            key = state2 + states1[i-1]
            transMat[ key ] = float(tokens[i])
    infile.close()
    
    if emissionProbs:
        return emittedSeq, hiddenSeq, transMat
    else:
        return hiddenSeq, transMat
    
def calcTransProb(hiddenSeq, transMat):
    prob = 0.5
    for i in range(len(hiddenSeq)-1):
        prob *= transMat[ hiddenSeq[i:i+2] ]
        
    return prob
    
# hiddenSeq, transMat = readTransitionFile('transitionProbs.txt')
# print calcTransProb(hiddenSeq, transMat)


def calcEmissionProb(emittedSeq, hiddenSeq, emissionMat):
    prob = 1.0
    for i in range(len(hiddenSeq)):
        prob *= emissionMat[ hiddenSeq[i] + emittedSeq[i] ]
        
    return prob
    
# emittedSeq, hiddenSeq, emissionMat = \
    # readTransitionFile('transitionProbs2.txt', emissionProbs=True)
# print calcEmissionProb(emittedSeq, hiddenSeq, emissionMat)

