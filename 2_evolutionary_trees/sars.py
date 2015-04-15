#!/usr/bin/env python

import math, os, random, re, sys


# get the key for the adjacency hash for nodes i,j
def getKey(i, j):
    return '%s:%s' % (i,j)
    
    
# update distances in the adjacency hash using the Floyd-Warshall algorithm
def updateAdjDistances(adjDict, i, j, k):
    if i==j or i==k or j==k: return
    
    try:
        dist = adjDict[getKey(i,k)] + adjDict[getKey(k,j)]
        if getKey(i,j) not in adjDict or adjDict[getKey(i,j)] > dist:
            adjDict[getKey(i,j)] = dist
    except:
        pass

        
# calculate the distance matrix given the node to node adjacency file
def getDistanceMatrix(adjFile):
    adjDict = dict()
    highestNode = 0
    
    # read adjacency information from file
    infile = open(adjFile, 'r')
    numLeaves = int(infile.readline())
    for line in infile:
        if line.startswith('#') or len(line.rstrip())==0:
            continue
        start, stop, weight = [ int(x) for x in re.split('->|:', line.rstrip()) ]
        adjDict[getKey(start, stop)] = weight
        highestNode = max(highestNode, start, stop)
    infile.close()
    
    # Calculate pairwise distances between nodes using the Floyd-Warshall algorithm
    for k in range(highestNode+1):
        for i in range(highestNode+1):
            for j in range(highestNode+1):
                updateAdjDistances(adjDict, i, j, k)
    
    # populate distance matrix
    distMat = [[0]*numLeaves for x in range(numLeaves)]
    for i in range(numLeaves):
        for j in range(numLeaves):
            if i!=j:
                distMat[i][j] = adjDict[getKey(i,j)]
    
    return distMat
    

distMat = getDistanceMatrix('adjToDistMat.txt')
print distMat

    