#!/usr/bin/env python

import math, os, random, re, sys



def readAdjList(infile):
    adjDict = dict()
    
    infile = open(infile, 'r')
    numLeaves = int(infile.readline())
    for line in infile:
        if line.startswith('#') or len(line.rstrip())==0:
            continue
        start, stop, weight = re.split('->|:', line.rstrip())
        adjDict[start] = adjDict.get(start, dict())
        adjDict[start][stop] = int(weight)
    infile.close()
    
    return adjDict, numLeaves
    

def getDist(adjDict, i, j, cumDist=0):
    if i==j: return 0
    
    # if i in adjDict and j in adjDict[i]:
        # dist = 
    
    return cumDist


def getDistMatrix(adjDict, numLeaves):
    distMat = [[0]*numLeaves for x in range(numLeaves)]
    
    for i in range(numLeaves):
        for j in range(numLeaves):
            distMat[i][j] = getDist(adjDict, i, j)
    
    return distMat

adjDict, numLeaves = readAdjList('adjToDistMat.txt')
print adjDict
distMat = getDistMatrix(adjDict, numLeaves)
print distMat

