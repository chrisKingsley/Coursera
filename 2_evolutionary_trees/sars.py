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
    
  
# class definition of a node in a binary tree of sequences  
class Node():
    def __init__(self):
        self.parent = None
        self.child1 = None
        self.child2 = None
        self.seq = ''
    
    def isLeaf(self):
        return self.child1 is None and \
               self.child2 is None
    
    def isHeadNode(self):
        return self.parent is None
        
    def addChild(self, child):
        if self.child1 is None:
            self.child1 = child
        elif self.child2 is None:
            self.child2 = child
        else:
            print 'adding child %s to node with two children' % \
                  child
            sys.exit()
            
    def __repr__(self):
        return 'parent:%s child1:%s child2:%s seq:%s' % \
               (self.parent, self.child1, self.child2, self.seq)
        
# read rooted tree from file       
def readRootedSeqTree(treeFile, headNode=True):
    infile = open(treeFile, 'r')
    numLeaves = int(infile.readline())
    tree = dict()
    leafNum = 0
    
    for line in infile:
        if re.search('[ACGT]', line):
            parentNode, seq = line.rstrip().split('->')
            node = Node()
            node.seq = seq
            node.parent = parentNode
            tree[ str(leafNum) ] = node
            if parentNode in tree:
                parent = tree[parentNode]
            else:
                parent = Node()
                tree[parentNode] = parent
            parent.addChild( str(leafNum) )
            leafNum += 1
        else:
            parentNode, childNode = line.rstrip().split('->')
            if parentNode in tree:
                node = tree[parentNode]
            else:
                node = Node()
                tree[parentNode] = node
            node.addChild(childNode)
                      
            if childNode in tree:
                node = tree[childNode]
            else:
                node = Node()
                tree[childNode] = node
            node.parent = parentNode
            
    infile.close()
    
    if headNode:
        for nodeNum in tree:
            if tree[nodeNum].isHeadNode():
                return tree, numLeaves, nodeNum
                
    return tree, numLeaves
    
    
# returns true if the passed node has no scores but both its
# children have scores
def nodeIsRipe(nodeNum, tree, scores):
    if len(scores[ nodeNum ]) > 0:
        return False
        
    node = tree[ nodeNum ]
    if len(scores[ node.child1 ])==0 or len(scores[ node.child2 ])==0:
        return False
        
    return True
    
    
# returns the scores for each possible base of the nodes in the passed
# tree at the given index in the sequences
def getSmallParsimonyScores(tree, numLeaves, charIdx):
    scores = { str(x):[] for x in range(len(tree)) }
    
    # per base score for leaves
    for nodeNum in range(numLeaves):
        node = tree[ str(nodeNum) ]
        for j in range(len(BASES)):
            dist = 0 if node.seq[charIdx]==BASES[j] else sys.maxint
            scores[ str(nodeNum) ].append( dist )
    
    # per base score for internal nodes
    ripeNodes = { str(x) for x in range(numLeaves, len(tree)) }
    while len(ripeNodes) > 0:
        nodesToSearch = list(ripeNodes)
        #print 'nodesToSearch:', nodesToSearch
        for nodeNum in nodesToSearch:
            
            if nodeIsRipe(nodeNum, tree, scores):
                node = tree[ nodeNum ]
                ripeNodes.remove(nodeNum)
                # print 'ripe node:%s child1:%s child2:%s' % (nodeNum, node.child1, node.child2)
                for j in range(len(BASES)):
                    child1_score = scores[ node.child1 ]
                    child2_score = scores[ node.child2 ]
                    baseScores1, baseScores2 = [], []
                    for k in range(len(BASES)):
                        distPenalty = 0 if j==k else 1
                        baseScores1.append(child1_score[k] + distPenalty)
                        baseScores2.append(child2_score[k] + distPenalty)
                    scores[nodeNum].append(min(baseScores1) + min(baseScores2))
    #print scores
    return scores


# returns the Hamming distance between the two passed sequences
def hammingDist(seq1, seq2):
    dist = 0
    for i in range(len(seq1)):
        if seq1[i]!=seq2[i]:
            dist += 1
    
    return dist
    
    
# updates the sequences in a node at the passed character
# index based on the passed scores
def updateNodeSeqs(tree, parentNodeNum, scores, charIdx):
    parentNode = tree[ parentNodeNum ]
    childNodeNums = [ parentNode.child1, parentNode.child2 ]
    
    for childNodeNum in childNodeNums:
        childNode = tree[ childNodeNum ]
        if childNode.isLeaf():
            continue
            
        baseScores = []
        for i in range(len(BASES)):
            dist = 0 if parentNode.seq[charIdx]==BASES[i] else 1
            baseScores.append( scores[ childNodeNum ][i] + dist )
        baseIdx = baseScores.index(min(baseScores))
        
        childNode.seq += BASES[baseIdx]
        updateNodeSeqs(tree, childNodeNum, scores, charIdx)

# update the sequences over all nodes in the tree at the specified
# index in the sequences
def updateTreeSeqs(tree, numLeaves, scores, charIdx, headNode):
    # assign character to root
    node = tree[ headNode ]
    baseIdx = scores[ headNode ].index(min(scores[ headNode ]))
    node.seq += BASES[baseIdx]

    # assign characters to internal nodes
    updateNodeSeqs(tree, headNode, scores, charIdx)
    
# returns the most parsimonious set of sequences for the internal
# nodes of the passed tree with the passed head node
def smallParsimony(tree, numLeaves, headNode):
    for charIdx in range(len(tree['0'].seq)):
        scores = getSmallParsimonyScores(tree, numLeaves, charIdx)
        updateTreeSeqs(tree, numLeaves, scores, charIdx, headNode)
        
    parsScore = 0
    for nodeNum in tree:
        node = tree[ nodeNum ]
        if not node.isLeaf():
            childNode1 = tree[ node.child1 ]
            childNode2 = tree[ node.child2 ]
            parsScore += hammingDist(node.seq, childNode1.seq) + \
                         hammingDist(node.seq, childNode2.seq)
    
    return tree, parsScore


# prints the sequences and Hamming distances of all nodes in the passed tree    
def printTree(tree, headNodeNum=None):
    for nodeNum in tree:
        node = tree[ nodeNum ]
        if headNodeNum is not None and nodeNum==headNodeNum:
            childNode1 = tree[ node.child1 ]
            childNode2 = tree[ node.child2 ]
            dist = hammingDist(childNode1.seq, childNode2.seq)
            print '%s->%s:%s' % (childNode1.seq, childNode2.seq, dist)
            print '%s->%s:%s' % (childNode2.seq, childNode1.seq, dist)
        elif not node.isLeaf():
            childNode1 = tree[ node.child1 ]
            childNode2 = tree[ node.child2 ]
            dist = hammingDist(node.seq, childNode1.seq)
            print '%s->%s:%s' % (node.seq, childNode1.seq, dist)
            print '%s->%s:%s' % (childNode1.seq, node.seq, dist)
            dist = hammingDist(node.seq, childNode2.seq)
            print '%s->%s:%s' % (node.seq, childNode2.seq, dist)
            print '%s->%s:%s' % (childNode2.seq, node.seq, dist)

# tree, numLeaves, headNode = readRootedSeqTree('treeSeqs.txt')
# tree, parsScore = smallParsimony(tree, numLeaves, headNode)
# print parsScore
# printTree(tree)


# adds internal nodes to unrooted tree with added head node
def addInternalNode(tree, parentNodeNum, childNodeNum, edges):
    if childNodeNum in tree:
        return
    
    newNode = Node()
    newNode.parent = parentNodeNum
    tree[ childNodeNum ] = newNode
    
    for edge in edges:
        node1, node2 = edge.split('->')
        if node1==childNodeNum and node2!=parentNodeNum:
            newNode.addChild( node2 )
            if newNode.child1 is not None and newNode.child2 is not None:
                break

    edges.remove('%s->%s' % (childNodeNum, newNode.child1))
    edges.remove('%s->%s' % (childNodeNum, newNode.child2))
    
    addInternalNode(tree, childNodeNum, newNode.child1, edges)
    addInternalNode(tree, childNodeNum, newNode.child2, edges)


# read unrooted tree from file and add head node
def readUnrootedSeqTree(treeFile):
    tree, edges = dict(), set()
    headNode, headNodeNum, leafNum = Node(), 0, 0
    
    # read adjacency list from file
    infile = open(treeFile, 'r')
    numLeaves = int(infile.readline())
    for line in infile:
        node1, node2 = line.rstrip().split('->')
        
        # add leaf nodes to tree
        if re.search('[ACGT]', node2):
            childNode = Node()
            childNode.seq = node2
            childNode.parent = node1
            tree[ str(leafNum) ] = childNode
            edges.add('%s->%s' % (node1, str(leafNum)))
            leafNum += 1
            
        # add internal edges and choose head node
        elif not re.search('[ACGT]', node1):
            edges.add( line.rstrip() )
            maxNode = max(int(node1), int(node2))
            if maxNode >= headNodeNum:
                headNodeNum = maxNode + 1
                headNode.child1 = node1
                headNode.child2 = node2
    infile.close()
    
    # add head node to tree
    headNodeNum = str(headNodeNum)
    tree[ headNodeNum ] = headNode
    edges.remove('%s->%s' % (headNode.child1, headNode.child2))
    edges.remove('%s->%s' % (headNode.child2, headNode.child1))
    
    # add internal nodes to tree
    addInternalNode(tree, headNodeNum, headNode.child1, edges)
    addInternalNode(tree, headNodeNum, headNode.child2, edges)
    
    return tree, numLeaves, headNodeNum
    
# tree, numLeaves, headNodeNum = readUnrootedSeqTree('unrootedTreeSeqs.txt')
# tree, parsScore = smallParsimony(tree, numLeaves, headNodeNum)
# print parsScore
# printTree(tree, headNodeNum=headNodeNum)


# reads the adjacency file for Nearest Neighbors of a Tree Problem
def readNN_adjList(fileName):
    tree = dict()
    
    infile = open(fileName, 'r')
    node1, node2 = infile.readline().strip().split()
    for line in infile:
        n1, n2 = line.rstrip().split('->')
        if n1 in tree:
            tree[ n1 ].append( n2 )
        else:
            tree[ n1 ] = [ n2 ]
    
    return tree, node1, node2
    
    
# print the passed Nearest Neighbors Tree 
def printNN_adjList(tree):
    for parent in tree:
        for child in tree[ parent ]:
            print '%s->%s' % (parent, child)
    print


# swaps the nodes around the passed nodes node1/node2
def swapNodes(tree, node1, node2, idx1, idx2):
    childNode1 = tree[ node1 ][ idx1 ]
    childIdx1 = tree[ childNode1 ].index(node1)
    tree[ childNode1 ][childIdx1] = node2
    
    childNode2 = tree[ node2 ][ idx2 ]
    childIdx2 = tree[ childNode2 ].index(node2)
    tree[ childNode2 ][childIdx2] = node1
    
    tree[ node1 ][idx1] = childNode2
    tree[ node2 ][idx2] = childNode1
    

# prints the two nearest neighbor trees by swapping subtrees
# adjacent to the two passed nodes
def printNN_trees(tree, node1, node2):
    nodeIdx1, nodeIdx2 = [],[]
    
    for i in range(3):
        if tree[ node1 ][i] != node2:
            nodeIdx1.append(i)
            
    for i in range(3):
        if tree[ node2 ][i] != node1:
            nodeIdx2.append(i)
    
    swapNodes(tree, node1, node2, nodeIdx1[0], nodeIdx2[0])
    printNN_adjList(tree)
    
    swapNodes(tree, node1, node2, nodeIdx1[0], nodeIdx2[1])
    printNN_adjList(tree)
    
    swapNodes(tree, node1, node2, nodeIdx1[0], nodeIdx2[0])
    # printNN_adjList(tree)
    
# tree, node1, node2 = readNN_adjList('NN_adjList.txt')
# printNN_trees(tree, node1, node2)


def removeHeadNode(tree, headNodeNum):
    newTree = dict()
    seqs = dict()
    
    # remove head node
    headNode = tree[ headNodeNum ]
    child1 = tree[ headNode.child1 ]
    child2 = tree[ headNode.child2 ]
    
    for node in tree:
        if node==headNodeNum:
            headNode = tree[ headNodeNum ]
            child1 = tree[ headNode.child1 ]
            child2 = tree[ headNode.child2 ]
        else:
            seqs[ node ] = tree[ node ].seq
            
    return newTree, seqs
    
    
tree, numLeaves, headNodeNum = readUnrootedSeqTree('largeParsimonyTree.txt')
tree, parsScore = smallParsimony(tree, numLeaves, headNodeNum)
tree, seqs = removeHeadNode(tree, headNodeNum)
print seqs
sys.exit()
print tree, numLeaves, headNodeNum

print tree, parsScore

