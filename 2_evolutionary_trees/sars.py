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
    
# distMat = getDistanceMatrix('adjToDistMat.txt')
# print distMat


# reads contents of file into a 2D distance matrix
def readDistMatFile(filePath, readNodeNum=False):
    infile = open(filePath, 'r')
    n = int(infile.readline().rstrip())
    if readNodeNum:
        j = int(infile.readline())
    distMat = []
    
    for i in range(n):
        vals = infile.readline().rstrip().split()
        distMat.append([ int(x) for x in vals ])
    infile.close()
    
    if readNodeNum:
        return distMat, j
    else:
        return distMat

        
# gets the limb length for a given node and distance matrix 
def getLimbLength(distMat, j):
    limbLength = sys.maxint
    
    for i in range(len(distMat)):
        for k in range(len(distMat)):
            if i==j or i==k or j==k:
                continue
            dist = (distMat[i][j]+distMat[j][k]-distMat[i][k])/2
            if dist < limbLength:
                limbLength = dist
                
    return limbLength

# distMat, j = readDistMatFile('distMat.txt', True)
# print getLimbLength(distMat, j)


def getAttachmentNodes(d, j):
    for i in range(j):
        for k in range(j):
            if i!=k and d[i][k]==d[i][j] + d[j][k]:
                return i,k

def attachNode(retVal, d, i, j, limbLength, numNodes):
    dist = d[i][j]
    print 'retVal2', retVal
    # find existing node at the correct ditance
    for k in range(j):
        if retVal.get('%s:%s' % (i, k), -1)==dist:
            retVal[ '%s:%s' % (k, j) ] = limbLength
            retVal[ '%s:%s' % (j, k) ] = limbLength
            return numNodes
            
    # add new node since exiting node not found
    print 'added node %d at %d between nodes %d & %d' % (numNodes,dist, i, k)
    retVal[ '%s:%s' % (i, numNodes) ] = dist
    retVal[ '%s:%s' % (numNodes, i) ] = dist
    retVal[ '%s:%s' % (k, numNodes) ] = d[i][k] - dist
    retVal[ '%s:%s' % (numNodes, k) ] = d[i][k] - dist
    retVal[ '%s:%s' % (j, numNodes) ] = limbLength
    retVal[ '%s:%s' % (numNodes, j) ] = limbLength
    if '%s:%s' % (i, k) in retVal: del retVal[ '%s:%s' % (i, k) ]
    if '%s:%s' % (k, i) in retVal: del retVal[ '%s:%s' % (k, i) ]
    
    return numNodes + 1

    
def addToMatrixRowCol(distMat, n, val):
    for i in range(n):
        if i!=n-1:
            distMat[i][n-1] = distMat[i][n-1]+val
            distMat[n-1][i] = distMat[i][n-1]


def additivePhylogeny(distMat, n, retVal=dict(), numNodes=None):
    if numNodes is None: numNodes = len(distMat)
    # exit recursion when only two nodes left
    if(n==2):
        retVal['0:1'] = distMat[0][1]
        retVal['1:0'] = distMat[1][0]
        return retVal, numNodes
        
    # create the bald tree, subtracting limb length of 
    # the last entry in the distance matrix
    limbLength = getLimbLength(distMat, n-1)
    addToMatrixRowCol(distMat, n, -limbLength)
    
    # get nodes between which node n should be placed
    i, k = getAttachmentNodes(distMat, n-1)
    print 'AttachmentNodes',i,k, ' dist', distMat[i][n-1]
    
    # call recursive function on shortened distance matrix
    retVal, numNodes = additivePhylogeny(distMat, n-1, retVal, numNodes)
    
    # regenerate the original distance matrix and add to tree
    addToMatrixRowCol(distMat, n, limbLength)
    numNodes = attachNode(retVal, distMat, i, n-1, limbLength, numNodes)
    
    return retVal, numNodes
    
distMat, j = readDistMatFile('distMat.txt', True)
tree, numNodes = additivePhylogeny(distMat, len(distMat))
for key in sorted(tree):
    print '%s->%s' % (key, tree[key])
 