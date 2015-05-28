#!/usr/bin/env python

import math, os, re, sys


def readProbTable(contents, lineNum):
    probMat = dict()
    
    states1 = contents[lineNum].split()
    lineNum +=1
    
    while lineNum < len(contents):
        if re.match('-+', contents[lineNum]):
            break
        tokens = contents[lineNum].split()
        state2 = tokens[0]
        probMat[ state2 ] = { states1[i-1]:float(tokens[i]) for i in range(1, len(tokens)) }
        lineNum += 1
            
    return probMat, lineNum + 1
        
        
def readTransitionFile(filePath, readEmittedSeq=True, readHiddenPath=False,
                       readTransitionProbs=True, readEmissionProbs=True):
    lineNum = 0
    infile = open(filePath, 'r')
    contents = [ x.rstrip() for x in infile.readlines() ]
    infile.close()
    
    if readEmittedSeq:
        emittedSeq = contents[lineNum]
        lineNum += 4
    
    if readHiddenPath:
        hiddenPath = contents[lineNum]
        lineNum += 4
    else:
        lineNum +=2
    
    if readTransitionProbs:
        transMat, lineNum = readProbTable(contents, lineNum)
    
    if readEmissionProbs:
        emissionMat, lineNum = readProbTable(contents, lineNum)
    
    retVal = []
    if readEmittedSeq: retVal.append( emittedSeq )
    if readHiddenPath: retVal.append( hiddenPath )
    if readTransitionProbs: retVal.append( transMat )
    if readEmissionProbs: retVal.append( emissionMat )
    return retVal
    
    
def calcTransProb(hiddenPath, transMat):
    prob = 0.5
    for i in range(len(hiddenPath)-1):
        prob *= transMat[ hiddenPath[i] ][ hiddenPath[i+1] ]
        
    return prob
    
# hiddenPath, transMat = readTransitionFile('transitionProbs.txt', False, True, True, False)
# print calcTransProb(hiddenPath, transMat)


def calcEmissionProb(emittedSeq, hiddenPath, emissionMat):
    prob = 1.0
    for i in range(len(hiddenPath)):
        prob *= emissionMat[ hiddenPath[i] ][ emittedSeq[i] ]
        
    return prob
    
# emittedSeq, hiddenPath, emissionMat = \
    # readTransitionFile('transitionProbs2.txt', True, True, False, True)
# print calcEmissionProb(emittedSeq, hiddenPath, emissionMat)


def getScoringMat(emittedSeq, emissionMat, transitionMat):
    hiddenStates = sorted(transitionMat.keys())
    stateProbs = [ [0]*len(hiddenStates) for x in range(len(emittedSeq)) ]
    
    # initialize first entry in state probability matrix
    stateProbs[0] = [1.0/len(hiddenStates)] * len(hiddenStates)
    for i in range(len(hiddenStates)):
        stateProbs[0][i] *= emissionMat[ hiddenStates[i] ][ emittedSeq[0] ]
    
    # complete the scoring matrix
    for i in range(1, len(emittedSeq)):
        for j in range(len(hiddenStates)):
            probs = []
            for k in range(len(hiddenStates)):
                prob = stateProbs[i-1][k] * \
                       transitionMat[ hiddenStates[k] ][ hiddenStates[j] ] *\
                       emissionMat[ hiddenStates[j] ][ emittedSeq[i] ]
                probs.append( prob )
            stateProbs[i][j] = max(probs)
            
    return stateProbs
    
    
def getHiddenPath(stateProbs, emissionMat, transitionMat, emittedSeq):
    hiddenStates = sorted(transitionMat.keys())
    
    # identify last state in the sequence of hidden states
    maxProbIdx = stateProbs[-1].index(max(stateProbs[-1]))
    hiddenPath = hiddenStates[ maxProbIdx ]
    
    # identify successive hidden states
    for i in range(len(stateProbs)-2, -1, -1):
        probs = []
        for j in range(len(hiddenStates)):
            prob = stateProbs[i][j] * \
                   transitionMat[ hiddenStates[j] ][ hiddenPath[0] ] * \
                   emissionMat[ hiddenPath[0] ][ emittedSeq[i+1] ]
            probs.append( prob )
        maxProbIdx = probs.index(max(probs))
        hiddenPath = hiddenStates[maxProbIdx] + hiddenPath
    
    return hiddenPath

def findOptimalHiddenPath(emittedSeq, emissionMat, transitionMat):
    stateProbs = getScoringMat(emittedSeq, emissionMat, transitionMat)
    hiddenPath = getHiddenPath(stateProbs, emissionMat, transitionMat, emittedSeq)
    
    return hiddenPath
    
    
emittedSeq, transitionMat, emissionMat = \
    readTransitionFile('transitionProbs3.txt', readEmissionProbs=True)
print findOptimalHiddenPath(emittedSeq, emissionMat, transitionMat)


