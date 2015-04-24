#!/usr/bin/env python

import math, os, random, re, sys


# get the key for the adjacency hash for nodes i,j
def getKey(i, j):
    return '%s:%s' % (i,j)
    
def inverseKey(key):
    i,j = key.split(':')
    return '%s:%s' % (j,i)
    
    
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
def getLimbLength(distMat, n, j):
    limbLength = sys.maxint
    
    for i in range(n):
        for k in range(n):
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

# def attachNode_old(tree, d, i, j, limbLength, numNodes):
    # dist = d[i][j]
    # print 'tree1', sorted(tree.items())
    # # find existing node at the correct distance
    # for k in range(j):
        # if tree.get(getKey(i, k), -1)==dist:
            # tree[ getKey(k, j) ] = limbLength
            # tree[ getKey(j, k) ] = limbLength
            # return numNodes
            
    # # add new node since exiting node not found
    # print 'added node %d at distance %d between nodes %d & %d' % \
        # (numNodes,dist, i, k)
    # tree[ getKey(i, numNodes) ] = dist
    # tree[ getKey(numNodes, i) ] = dist
    # tree[ getKey(k, numNodes) ] = d[i][k] - dist
    # tree[ getKey(numNodes, k) ] = d[i][k] - dist
    # tree[ getKey(j, numNodes) ] = limbLength
    # tree[ getKey(numNodes, j) ] = limbLength
    # if getKey(i, k) in tree: del tree[ getKey(i, k) ]
    # if getKey(k, i) in tree: del tree[ getKey(k, i) ]
    
    # return numNodes + 1
    
    
# def getPath (tree, i, k, path):
    # nodes = [ x for x in tree.keys() if x.startswith(str(i)+":") ]
    # print "i:%s k:%s path:%s nodes:%s" % (i,k,path,nodes)
    # for node in nodes:
        # print 'path:%s node:%s' % (path, node)
        # if inverseKey(node) not in path:
            # path.append(node)
            # print 'node', node
            # if node.endswith(':'+str(k)):
                # print 'final path:%s last node:%s' % (path, node)
                # return path
            # l = int(node.split(':')[1])
            # getPath(tree, l, k, path)
            # path.pop()
    
    
def getPathInTree(tree, i, k, path):
    if len(path)>0 and path[-1].endswith(':'+str(k)):
        return path
    
    nodes = [ x for x in tree.keys() if x.startswith(str(i)+":") ]
    print "i:%s k:%s path:%s nodes:%s" % (i,k,path,nodes)
    for node in nodes:
        if inverseKey(node) not in path:
            path.append(node)
            l = int(node.split(':')[1])
            path = getPathInTree(tree, l, k, path)
            if not path[-1].endswith(':'+str(k)):
                path.pop()
            
    return path
    
    
def attachNode(tree, d, i, j, k, dist, limbLength, numNodes):
    path = getPathInTree(tree, i, k, [])
    print 'final path:%s' % path
    print 'tree1', sorted(tree.items())
    
    for node in path:
        i, j =[ int(x) for x in node.split(':') ]
        print 'node:%s dist:%s' % (node, dist)
        if dist==tree[node]:
            tree[ getKey(k, j) ] = limbLength
            tree[ getKey(j, k) ] = limbLength
            break
        elif dist < tree[node]:
            numNodes += 1
            tree[ getKey(i, numNodes) ] = dist
            tree[ getKey(numNodes, i) ] = dist
            tree[ getKey(k, numNodes) ] = d[i][k] - dist
            tree[ getKey(numNodes, k) ] = d[i][k] - dist
            tree[ getKey(j, numNodes) ] = limbLength
            tree[ getKey(numNodes, j) ] = limbLength
            if getKey(i, k) in tree: del tree[ getKey(i, k) ]
            if getKey(k, i) in tree: del tree[ getKey(k, i) ]
            break
            
        dist = dist - tree[node]
    
    return numNodes + 1

    
def addToMatrixRowCol(distMat, n, val):
    for i in range(n):
        if i!=n-1:
            distMat[i][n-1] = distMat[i][n-1]+val
            distMat[n-1][i] = distMat[i][n-1]


def additivePhylogeny(distMat, n, tree=dict(), numNodes=None):
    if numNodes is None: numNodes = len(distMat)
    # exit recursion when only two nodes left
    if(n==2):
        tree['0:1'] = distMat[0][1]
        tree['1:0'] = distMat[1][0]
        return tree, numNodes
        
    # create the bald tree, subtracting limb length of 
    # the last entry in the distance matrix
    limbLength = getLimbLength(distMat, n, n-1)
    print 'node %d limbLength %0.2f' % (n-1, limbLength)
    addToMatrixRowCol(distMat, n, -limbLength)
    print distMat
    
    # call recursive function on reduced distance matrix
    tree, numNodes = additivePhylogeny(distMat, n-1, tree, numNodes)
    
    # get nodes between which node n should be placed and add node to tree
    i, k = getAttachmentNodes(distMat, n-1)
    print 'AttachmentNodes',i,k, ' dist', distMat[i][n-1]
    numNodes = attachNode(tree, distMat, i, n-1, k, distMat[i][n-1], limbLength, numNodes)
    print 'tree2', sorted(tree.items())
    # regenerate the original distance matrix 
    addToMatrixRowCol(distMat, n, limbLength)
    
    return tree, numNodes
    
distMat, j = readDistMatFile('distMat.txt', True)
tree, numNodes = additivePhylogeny(distMat, len(distMat))
for key in sorted(tree):
    print '%s->%s' % (key, tree[key])
 