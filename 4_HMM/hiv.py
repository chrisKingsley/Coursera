#!/usr/bin/env python

import math, os, re, sys


# read table of transition or emission probabilites from array of
# file contents
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
        
   
# read  file containing transition and/or emission probabilities
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
    
    
# given hidden path and transition scoreing matrix, calculate the
# overall transition probability
def calcTransProb(hiddenPath, transMat):
    prob = 0.5
    for i in range(len(hiddenPath)-1):
        prob *= transMat[ hiddenPath[i] ][ hiddenPath[i+1] ]
        
    return prob
    
# hiddenPath, transMat = readTransitionFile('transitionProbs.txt', False, True, True, False)
# print calcTransProb(hiddenPath, transMat)


# calculates the emission probablility of the emitted sequence
# given the hidden path and the emission scoring matrix
def calcEmissionProb(emittedSeq, hiddenPath, emissionMat):
    prob = 1.0
    for i in range(len(hiddenPath)):
        prob *= emissionMat[ hiddenPath[i] ][ emittedSeq[i] ]
        
    return prob
    
# emittedSeq, hiddenPath, emissionMat = \
    # readTransitionFile('transitionProbs2.txt', True, True, False, True)
# print calcEmissionProb(emittedSeq, hiddenPath, emissionMat)


# returns the scoring matrix for determining the most probable
# hidden state given an emitted sequence and the transition/emission
# scoring matrices of an HMM
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
    
    
# returns the most probable hidden path of the emitted sequence, given
# the max state probabilities at each position and transition/emission 
# matrices
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

    
# find the most probable hidden path given the emitted sequence,
# and emission/transition scoring matrices
def findOptimalHiddenPath(emittedSeq, emissionMat, transitionMat):
    stateProbs = getScoringMat(emittedSeq, emissionMat, transitionMat)
    hiddenPath = getHiddenPath(stateProbs, emissionMat, transitionMat, emittedSeq)
    
    return hiddenPath
    
# emittedSeq, transitionMat, emissionMat = \
    # readTransitionFile('transitionProbs3.txt', readEmissionProbs=True)
# print findOptimalHiddenPath(emittedSeq, emissionMat, transitionMat)


# returns the likelihood matrix for the probabilities associated with
# all the hidden states of the emitted sequence at each position
def getLikelihoodMatrix(emittedSeq, emissionMat, transitionMat):
    hiddenStates = sorted(transitionMat.keys())
    stateProbs = [ [0]*len(hiddenStates) for x in range(len(emittedSeq)) ]
    
    # initialize first entry in state probability matrix
    stateProbs[0] = [1.0/len(hiddenStates)] * len(hiddenStates)
    for i in range(len(hiddenStates)):
        stateProbs[0][i] *= emissionMat[ hiddenStates[i] ][ emittedSeq[0] ]
        
    # complete the scoring matrix
    for i in range(1, len(emittedSeq)):
        for j in range(len(hiddenStates)):
            prob = 0.0
            for k in range(len(hiddenStates)):
                prob += stateProbs[i-1][k] * \
                        transitionMat[ hiddenStates[k] ][ hiddenStates[j] ] * \
                        emissionMat[ hiddenStates[j] ][ emittedSeq[i] ]
            stateProbs[i][j] = prob
            
    return stateProbs

    
# determine the likelihood that a sequence was emitted by an HMM, given
# the emitted sequence and the transition/emission probabilities of the HMM
def outcomeLikelihood(emittedSeq, emissionMat, transitionMat):
    stateProbs = getLikelihoodMatrix(emittedSeq, emissionMat, transitionMat)
    
    return sum(stateProbs[-1])
    
# emittedSeq, transitionMat, emissionMat = \
    # readTransitionFile('transitionProbs4.txt', readEmissionProbs=True)
# print outcomeLikelihood(emittedSeq, emissionMat, transitionMat)


# reads file containing alignments and parameters to build an HMM
def readAlignmentFile(filePath):
    lineNum = 0
    infile = open(filePath, 'r')
    contents = [ x.rstrip() for x in infile.readlines() ]
    infile.close()
    
    theta = float(contents[lineNum])
    lineNum += 2
    alphabet = contents[lineNum].split()
    lineNum += 2
    
    seqs = []
    for i in range(lineNum, len(contents)):
        seqs.append(contents[i])
        
    return theta, alphabet, seqs
    
    
def fracIndel(seqs, pos):
    numIndel = 0.0
    for i in range(len(seqs)):
        if seqs[i][pos]=='-':
            numIndel += 1.0
            
    return numIndel/len(seqs)

    
def addProbs(probHash, key1, key2, val):
    if key1 not in probHash:
        probHash[ key1 ] = dict()
    probHash[ key1 ][ key2 ] = val

    
def processInsertion(seqs, emittedMat, pos):
    insertionFound = False
    startPos = pos
    
    while pos < len(seqs[0]) and fracIndel(seqs, pos) > theta:
        insertionFound = True
        pos += 1
            
    if insertionFound:
        emittedVals = []
        
        for i in range(len(seqs)):
            if re.match('^-+$', seqs[i][startPos:pos]):
                emittedVals.append('-'*(pos-startPos))
            else:
                emittedVals.append( seqs[i][startPos:pos] )
                
        emittedMat.append(emittedVals)
    else:
        emittedMat.append( ['-']*len(seqs) )
        
    return pos
    
    
def getAlignmentMat(seqs, theta):
    emittedMat = []
    
    # check first position for insertion
    pos = processInsertion(seqs, emittedMat, 0)
    
    while pos < len(seqs[0]):
        emittedVals = []
        for i in range(len(seqs)):
            emittedVals.append(seqs[i][pos])
        emittedMat.append(emittedVals)
        pos += 1
        
        pos = processInsertion(seqs, emittedMat, pos)
        
    return emittedMat

def containsInsertion(vals):
    for val in vals:
        if val!='-':
            return True
    return False

def getAlignProbs(emittedMat, transProbs, emitProbs, i):
    stateNum = i/2 + 1
    mKey, dKey, iKey = 'M%d' % stateNum, 'D%d' % stateNum, 'I%d' % stateNum
    mKey2, dKey2 = 'M%d' % (stateNum+1), 'D%d' % (stateNum+1)
    
    # transition probabilities
    numEmits, numDels = 0.0, 0.0
    transProbs[ mKey ], transProbs[ dKey ] = dict(), dict()
    for j in range(len(emittedMat[i])):    
        if emittedMat[i][j]=='-':
            numDels += 1
            key = dKey
        else:
            numEmits += 1
            key = mKey
            
        if not re.match('^-+$', emittedMat[i+1][j]):
            key2 = iKey
        elif i==len(emittedMat)-2:
            key2 = 'E'
        elif emittedMat[i+2][j]=='-':
            key2 = dKey2
        else:
            key2 = mKey2
            
        transProbs[ key ][ key2 ] = transProbs[ key ].get(key2, 0) + 1
        
    for key2 in transProbs[ mKey ]:
        transProbs[ mKey ][ key2 ] /= numEmits
    for key2 in transProbs[ dKey ]:
        transProbs[ dKey ][ key2 ] /= numDels
        
    
    # emission probabilities
    emitProbs[ mKey ] = dict()
    numEmits = 0.0
    for j in range(len(emittedMat[i])):
        if emittedMat[i][j]!='-':
            numEmits+=1
            emitProbs[ mKey ][ emittedMat[i][j] ] = \
                emitProbs[ mKey ].get(emittedMat[i][j], 0) + 1
    if numEmits>0:
        for key2 in emitProbs[ mKey ]:
            emitProbs[ mKey ][ key2 ] /= numEmits
    
def getAlignInsertProbs(emittedMat, transProbs, emitProbs, alphabet, i):
    key = 'I%d' % (i/2)
    emitProbs[ key ] = dict()
    
    # transition probabilities
    if i==0:
        fracInsert = 1 - emittedMat[0].count('-')/float(len(emittedMat[0]))
        addProbs(transProbs, 'S', 'I0', fracInsert)
        addProbs(transProbs, 'S', 'M1', 1-fracInsert)
    if containsInsertion(emittedMat[i]):
        transProbs[ key ] = dict()
        if i==len(emittedMat)-1: 
            addProbs(transProbs, key, 'E', 1.0)
        else:
            nextStateNum, numInserts = i/2 + 1, 0.0
            for j in range(len(emittedMat[i])):
                if not re.match('^-+$', emittedMat[i][j]):
                    numInserts += 1
                    if(emittedMat[i+1][j]=='-'):
                        key2 = 'D%d' % nextStateNum
                    else:
                        key2 = 'M%d' % nextStateNum
                    transProbs[ key ][ key2 ] = \
                            transProbs[ key ].get(key2, 0) + 1
                    
                    numReinsert = len(emittedMat[i][j])-emittedMat[i][j].count('-')-1
                    if numReinsert > 0:
                        key2 = key
                        transProbs[ key ][ key2 ] = \
                            transProbs[ key ].get(key2, 0) + numReinsert
                        numInserts += numReinsert
                    
            for key2 in transProbs[ key ]:
                transProbs[ key ][ key2 ] /= numInserts
            print transProbs
            
    
    # emission probabilities
    numInserts=0.0
    for j in range(len(emittedMat[i])):
        if not re.match('^-+$', emittedMat[i][j]):
            for k in emittedMat[i][j]:
                if k in alphabet:
                    emitProbs[ key ][ k ] = \
                        emitProbs[ key ].get(k, 0) + 1
                    numInserts+=1
    if numInserts>0:
        for key2 in emitProbs[ key ]:
            emitProbs[ key ][ key2 ] /= numInserts
        
    
    
def getProbMats(emittedMat):
    transProbs, emitProbs = dict(), dict()
    print emittedMat
    
    # emission probs
    for i in range(len(emittedMat)):
        if i % 2 == 0:
            getAlignInsertProbs(emittedMat, transProbs, emitProbs, alphabet, i)
        else:
            getAlignProbs(emittedMat, transProbs, emitProbs, i)
    
    return transProbs, emitProbs
    
def printProbMats(emittedMat, transProbs, emitProbs, alphabet):
    states = ['S', 'I0']
    for i in range(len(emittedMat)/2):
        for state in ['M','D','I']:
            states.append( '%s%d' % (state, i+1) )
    states.append('E')
    
    # transmission probs
    print '\t%s' % '\t'.join(states)
    for state1 in states:
        print state1,
        for state2 in states:
            try:
                print '\t%0.3f' % transProbs[state1][state2],
            except:
                print '\t0',
        print
    
    # emission probs
    print '--------\n\t%s' % '\t'.join(alphabet)
    for state in states:
        print state,
        for emitChar in alphabet:
            try:
                print '\t%0.3f' % emitProbs[state][emitChar],
            except:
                print '\t0',
        print
        
def makeProbMatsFromAlignment(theta, alphabet, seqs):
    emittedMat = getAlignmentMat(seqs, theta)
    transProbs, emitProbs = getProbMats(emittedMat)
    
    printProbMats(emittedMat, transProbs, emitProbs, alphabet)
            
            

theta, alphabet, seqs = readAlignmentFile('alignment.txt')
makeProbMatsFromAlignment(theta, alphabet, seqs)
