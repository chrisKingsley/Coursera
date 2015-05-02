#!/usr/bin/env python

import math, os, random, re, sys

BASES = ['A','C','G','T']


# get the key for the adjacency hash for nodes i,j
def getKey(i, j):
    return '%s:%s' % (i,j)
# reverse the order of the passed key 
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


#  returns the nodes that the node j should be inserted between
def getAttachmentNodes(d, j):
    for i in range(j):
        for k in range(j):
            if i!=k and d[i][k]==d[i][j] + d[j][k]:
                return i,k
  

# returns an array containing the path from node i to k in the tree
def getPathInTree(tree, i, k, path):
    if len(path)>0 and path[-1].endswith(':'+str(k)):
        return path
    
    nodes = [ x for x in tree.keys() if x.startswith(str(i)+":") ]
    for node in nodes:
        if len(path)==0 or inverseKey(node)!=path[-1]:
            path.append(node)
            l = int(node.split(':')[1])
            path = getPathInTree(tree, l, k, path)
            if not path[-1].endswith(':'+str(k)):
                path.pop()
            
    return path
    

# attaches the node j to the tree, by adding a new node between two
# existing nodes or by attaching to an existing node, if possible
def attachNode(tree, d, i, j, k, dist, limbLength, numNodes):
    path = getPathInTree(tree, i, k, [])
    
    for node in path:
        i, k = [ int(x) for x in node.split(':') ]
        if dist==tree[node]:
            tree[ getKey(k, j) ] = limbLength
            tree[ getKey(j, k) ] = limbLength
            break
        elif dist < tree[node]:
            tree[ getKey(i, numNodes) ] = dist
            tree[ getKey(numNodes, i) ] = dist
            tree[ getKey(k, numNodes) ] = tree[node] - dist
            tree[ getKey(numNodes, k) ] = tree[node] - dist
            tree[ getKey(j, numNodes) ] = limbLength
            tree[ getKey(numNodes, j) ] = limbLength
            del tree[ getKey(i, k) ]
            del tree[ getKey(k, i) ]
            numNodes += 1
            break
            
        dist = dist - tree[node]
    
    return numNodes

    
# adds the passed value to the row/column index at position n
# in the passed distance matrix
def addToMatrixRowCol(distMat, n, val):
    for i in range(n):
        if i!=n-1:
            distMat[i][n-1] = distMat[i][n-1]+val
            distMat[n-1][i] = distMat[i][n-1]


# for the given additive distance matrix, returns a tree of
# nodes and their distances, and the total number of nodes
# in the tree
def additivePhylogeny(distMat, n, tree=dict(), numNodes=None):
    if numNodes is None: numNodes = len(distMat)
    # exit recursion when only two nodes left
    if(n==2):
        tree['0:1'] = distMat[0][1]
        tree['1:0'] = distMat[1][0]
        return tree, numNodes
        
    # create the bald tree, subtracting limb length of the 
    # last entry in the distance matrix from its rows/cols
    limbLength = getLimbLength(distMat, n, n-1)
    addToMatrixRowCol(distMat, n, -limbLength)
    
    # call recursive function on reduced distance matrix
    tree, numNodes = additivePhylogeny(distMat, n-1, tree, numNodes)
    
    # get nodes between which node n should be placed and add node to tree
    i, k = getAttachmentNodes(distMat, n-1)
    numNodes = attachNode(tree, distMat, i, n-1, k, distMat[i][n-1], limbLength, numNodes)

    # regenerate the original distance matrix 
    addToMatrixRowCol(distMat, n, limbLength)
    
    return tree, numNodes
    
# distMat, j = readDistMatFile('distMat.txt', True)
# tree, numNodes = additivePhylogeny(distMat, len(distMat))
# for key in sorted(tree):
    # print '%s:%s' % (re.sub(':','->', key), tree[key])


# returns the distance between two clusters of leaves
# from the passed distance matrix
def getClusterDist(distMat, leaves1, leaves2):
    dist = 0.0
    
    for leaf1 in leaves1:
        for leaf2 in leaves2:
            dist += distMat[leaf1][leaf2]
            
    return dist/(len(leaves1)*len(leaves2))
    
# returns the cluster ID's and distance between the two
# closest clusters of leaves
def closestClusters(distMat, clusters):
    clusterId1, clusterId2, minDist = 0, 0, sys.maxint
    clusterIds = clusters.keys()
    
    for i in range(len(clusterIds)-1):
        leaves1 = clusters[ clusterIds[i] ]
        for j in range(i+1, len(clusterIds)):
            leaves2 = clusters[ clusterIds[j] ]
            dist = getClusterDist(distMat, leaves1, leaves2)
            if dist < minDist:
                minDist = dist
                clusterId1 = clusterIds[i]
                clusterId2 = clusterIds[j]
                
    return clusterId1, clusterId2, minDist
    

# merges the two closest clusters of leaves, and updates ages
# of internal nodes
def mergeClusters(tree, distMat, clusters, ages):
    i, j, minDist = closestClusters(distMat, clusters)
    newCluster = clusters[i] | clusters[j]
    clusterNum = max(clusters.keys()) + 1
    
    clusters[ clusterNum ] = newCluster
    del clusters[ i ]
    del clusters[ j ]
    
    tree[ getKey(i,clusterNum) ] = minDist/2.0 - ages.get(i, 0)
    tree[ getKey(clusterNum,i) ] = tree[ getKey(i,clusterNum) ]
    tree[ getKey(j,clusterNum) ] = minDist/2.0 - ages.get(j, 0)
    tree[ getKey(clusterNum,j) ] = tree[ getKey(j,clusterNum) ]
    
    ages[ clusterNum ] = tree[ getKey(i,clusterNum) ] + ages.get(i, 0)


# Implements UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
# for the passed distance matrix
def upgma(distMat):
    tree, ages = dict(), dict()
    clusters = { x:{x} for x in range(len(distMat)) } # dict of dicts of leaves
    
    while len(clusters) > 1:
        mergeClusters(tree, distMat, clusters, ages)
    
    return tree
     
# distMat = readDistMatFile('distMat2.txt', False)
# tree = upgma(distMat)
# for key in sorted(tree):
    # print '%s:%s' % (re.sub(':','->', key), tree[key])


# uses neighbor joining to identify row/col indices
# of two neighboring leaves
def getNeighbors(distMat):
    id1, id2, minDist = 0, 0, sys.maxint
    totalDist = [ sum(distMat[i]) for i in range(len(distMat)) ]
    
    for i in range(len(distMat)-1):
        for j in range(i+1, len(distMat)):
            dist = (len(distMat)-2) * distMat[i][j] - \
                    totalDist[i] - totalDist[j]
            if dist < minDist:
                id1, id2, minDist = i, j, dist
    
    delta = (totalDist[id1] - totalDist[id2])/(len(distMat)-2.0)
    
    return id1, id2, delta
    
    
# takes the two neighboring leaves identified in the distance matrix
# and merges them into one new node, thereby reducing the dimension
# of the distance matrix
def reduceMatrix(distMat, labels, i, j):
    idx = [ x for x in range(len(distMat)) if x!=i and x!=j ]
    
    # create new label for merged node
    newLabel = max(labels) + 1
    newLabels = [ labels[x] for x in idx ]
    newLabels.append(newLabel)
    
    # allocate new distance matrix and copy values
    newMat = [[0]*len(newLabels) for x in range(len(newLabels)) ]
    for k in range(len(idx)):
        for l in range(len(idx)):
            newMat[k][l] = distMat[ idx[k] ][ idx[l] ]
            
    # calculate distance from new nodes to old nodes
    m = len(newMat)-1
    for k in range(len(idx)):
        newMat[m][k] = 0.5*(distMat[idx[k]][i] + distMat[idx[k]][j] - \
                            distMat[i][j])
        newMat[k][m] = newMat[m][k]
    
    return newMat, newLabels
   

# adds edges to the new node in the tree, connecting to existing nodes   
def addEdges(tree, distMat, labels, newLabel, i, j, delta):
    m = len(distMat)-1
    
    tree[ getKey(labels[i],newLabel) ] = (distMat[i][j] + delta)/2.0
    tree[ getKey(newLabel,labels[i]) ] = tree[ getKey(labels[i],newLabel) ]
    tree[ getKey(labels[j],newLabel) ] = (distMat[i][j] - delta)/2.0
    tree[ getKey(newLabel,labels[j]) ] = tree[ getKey(labels[j],newLabel) ]

    
# use the neighbor joining algorithm to construct a tree based on the
# passed distance matrix
def neighborJoining(distMat, labels, tree=dict()):
    if len(distMat)==2:
        tree[ getKey(labels[0],labels[1]) ] = distMat[0][1]
        tree[ getKey(labels[1],labels[0]) ] = distMat[1][0]
        return tree
        
    i, j, delta = getNeighbors(distMat)
    newMat, newLabels = reduceMatrix(distMat, labels, i, j)
    
    tree = neighborJoining(newMat, newLabels, tree)
    addEdges(tree, distMat, labels, newLabels[-1], i, j, delta)
    
    return tree
    
# distMat = readDistMatFile('distMat3.txt', False)
# tree = neighborJoining(distMat, range(len(distMat)))
# for key in sorted(tree):
    # print '%s:%s' % (re.sub(':','->', key), tree[key])
    
    
def readSeqTree(treeFile):
    infile = open(treeFile, 'r')
    numLeaves = int(infile.readline())
    seqTree = ['']*(2*numLeaves)
    
    for line in infile:
        parentNode, seq = line.rstrip().split('->')
        if re.search('[ACGT]', seq):
            parentNode = int(parentNode) % numLeaves + numLeaves/2
            if not seqTree[2*parentNode]:
                seqTree[2*parentNode] = seq
            else:
                seqTree[2*parentNode+1] = seq

    infile.close()
    
    return seqTree, numLeaves
    
    
def getSmallParsimonyScores(tree, numLeaves, charIdx):
    scores = [[] for x in range(2*numLeaves) ]
    
    # per base score for leaves
    for i in range(numLeaves, 2*numLeaves):
        for j in range(len(BASES)):
            dist = 0 if tree[i][charIdx]==BASES[j] else sys.maxint
            scores[i].append( dist )
    
    # per base score for internal nodes
    for i in range(numLeaves-1, 0, -1):
        for j in range(len(BASES)):
            childScores1, childScores2 = [], []
            for k in range(len(BASES)):
                distPenalty = 0 if j==k else 1
                childScores1.append(scores[2*i][k] + distPenalty)
                childScores2.append(scores[2*i+1][k] + distPenalty)
            scores[i].append(min(childScores1) + min(childScores2))
    
    return scores


def hammingDist(seq1, seq2):
    dist = 0
    for i in range(len(seq1)):
        if seq1[i]!=seq2[i]:
            dist += 1
    
    return dist
    
    
def updateTreeSeqs(tree, numLeaves, scores, charIdx):
    # assign character to root
    baseIdx = scores[1].index(min(scores[1]))
    tree[1] += BASES[baseIdx]
    
    # assign characters to internal nodes
    for i in range(2, numLeaves):
        baseScores = []
        for j in range(len(BASES)):
            dist = 0 if tree[i/2][charIdx]==BASES[j] else 1
            baseScores.append( scores[i][j] + dist )
            # baseScores.append( scores[i][j] )
        baseIdx = baseScores.index(min(baseScores))
        tree[i] += BASES[baseIdx]
    
    
def smallParsimony(tree, numLeaves):
    for charIdx in range(len(tree[-1])):
        scores = getSmallParsimonyScores(tree, numLeaves, charIdx)
        #print scores
        updateTreeSeqs(tree, numLeaves, scores, charIdx)
        
    parsScore = 0
    for i in range(1,numLeaves):
        parsScore += hammingDist(tree[i], tree[2*i]) + \
                     hammingDist(tree[i], tree[2*i+1])
    
    return tree, parsScore

tree, n = readSeqTree('treeSeqs.txt')
tree, parsScore = smallParsimony(tree, n)
print parsScore
for i in range(1, n):
    dist = hammingDist(tree[i], tree[2*i])
    print '%s->%s:%s' % (tree[i], tree[2*i], dist)
    print '%s->%s:%s' % (tree[2*i], tree[i], dist)
    dist = hammingDist(tree[i], tree[2*i+1])
    print '%s->%s:%s' % (tree[i], tree[2*i+1], dist)
    print '%s->%s:%s' % (tree[2*i+1], tree[i], dist)
