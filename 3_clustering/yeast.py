#!/usr/bin/env python

import math, re, sys


# reads a matrix of points.  Reads centers if the
# passed distortion flag is true
def readMatrixFile(fileName, distortion=False, betaParam=False):
    matrix, centers, readCenters = [], [], True
    
    infile = open(fileName, 'r')
    k, m = [ int(x) for x in infile.readline().rstrip().split() ]
    if betaParam:
        beta = float(infile.readline().rstrip())
    
    for line in infile:
        if distortion and re.match('-+', line):
            readCenters = False
            continue
        
        vals = [ float(x) for x in line.rstrip().split() ]
        if readCenters and distortion:
            centers.append(vals)
        else:
            matrix.append(vals)
    infile.close()
    
    if distortion:
        return  k, m, matrix, centers
    elif betaParam:
        return k, m, beta, matrix
    else:
        return  k, m, matrix
    
    
# returns the Euclidean distance between two lists
def euclideanDist(list1, list2):
    dist = 0.0
    for i in range(len(list1)):
        dist += (list1[i]-list2[i])*(list1[i]-list2[i])
    
    return math.sqrt(dist)

    
# returns the dot product between two lists
def dotProd(list1, list2):
    prod = 0.0
    for i in range(len(list1)):
        prod += (list1[i]*list2[i])
    
    return prod
    
# for the given set of points and value of k, returns the
# center points that have the largest distance to the
# closest center
def farthestFirstTraversal(k, matrix):
    centers = [ matrix[0] ]
    
    while len(centers) < k:
        maxDist, maxIdx = -sys.maxint, 0
        for i in range(1, len(matrix)):
            dists = []
            for j in range(len(centers)):
                dists.append(euclideanDist(matrix[i], centers[j]))
            dist = min(dists)
            if dist > maxDist:
                maxDist, maxIdx = dist, i
        centers.append( matrix[maxIdx] )
    
    return centers

# k, m, data = readMatrixFile('geMatrix.txt')
# centers = farthestFirstTraversal(k, data)
# for x in centers:
    # print ' '.join([ str(y) for y in x ])
    

# for the given set of data of points and centers, returns the 
# squared error distortion (mean squared minimum distance between
# each point and all centers)
def squaredErrorDistortion(data, centers):
    distortionDist = 0.0
    
    for i in range(len(data)):
        closestDist = sys.maxint
        for j in range(len(centers)):
            dist = euclideanDist(data[i],centers[j])
            if dist < closestDist:
                closestDist = dist
        distortionDist += closestDist*closestDist
        
    return distortionDist/len(data)

# k, m, data, centers = readMatrixFile('distortionMat.txt', distortion=True)
# print squaredErrorDistortion(data, centers)


# for the passed data points and centers, assigns each point
# to its closest center.  Returns a dict where center number
# points to the indices of the points in data that are in 
# the cluster with that center
def assignClusters(data, centers, k):
    clusters = { x:[] for x in range(k) }
    
    for i in range(len(data)):
        closestDist, centerIdx = sys.maxint, 0
        for j in range(k):
            dist = euclideanDist(data[i],centers[j])
            if dist < closestDist:
                closestDist, centerIdx  = dist, j
                
        clusters[ centerIdx ].append( i )
    
    return clusters

    
# for the given set of points and their dimension m, returns
# the center of gravity (mean of the sum of each component)
def centerOfGravity(points, m):
    centers = [0.0]*m
    
    for i in range(len(points)):
        for j in range(m):
            centers[j] += points[i][j]/len(points)
            
    return centers


# for the given set of data, clusters, k (# of clusters), and
# m (dimension of points), determines the center of each cluster
# and returns in a 2D array
def getCenters(data, clusters, k, m):
    centers = []
    
    for i in clusters:
        points = []
        for j in clusters[ i ]:
            points.append( data[j] )
        center = centerOfGravity(points, m)
        centers.append( center )
            
    return centers
    
    
# clusters the set of m dimensional points in data
# into k clusters using Lloyd's algorithm
def kMeansClustering(data, k, m):
    centers = []
    for i in range(k):
        centers.append( data[i] )
        
    while True:
        clusters = assignClusters(data, centers, k)
        newCenters = getCenters(data, clusters, k, m)
        if newCenters==centers:
            break
        centers = newCenters
        
    return centers
    
# k, m, data = readMatrixFile('kMeansMatrix.txt')
# centers = kMeansClustering(data, k, m)
# for center in centers:
    # print ' '.join([ '%0.3f' % x for x in center ])

    
# expectation step in EM algorithm for soft k-means clustering.
# estimates new responsibility matrix based on cluster centers
def eStep(data, beta, centers):
    resp = [ [0]*len(data) for x in range(len(centers)) ]
    
    # calculate responsibility of each cluster for each data point
    for i in range(len(data)):
        sum = 0.0
        for j in range(len(centers)):
            dist = euclideanDist(centers[j], data[i])
            resp[j][i] = math.exp(-beta * dist)
            sum += resp[j][i]
        for j in range(len(centers)):
            resp[j][i] /= sum
            
    return resp
    
    
# maximum likelihood step for EM algorithm for soft k-means clustering.
# estimates new cluster centers based on responsibility matrix
def mStep(data, respMatrix, m):
    centers = []
    
    # calculate new centers based on data and responsibility matrix
    for i in range(len(respMatrix)):
        center = []
        for j in range(m):
            dotProd, denom = 0.0, 0.0
            for k in range(len(data)):
                dotProd += data[k][j]*respMatrix[i][k]
                denom += respMatrix[i][k]
            center.append(dotProd/denom)
        centers.append(center)
        
    return centers

    
# soft k-means algorithm that uses EM
def soft_kMeans(data, k, m, beta, numIter=100):
    centers = [ data[x] for x in range(k) ]
    
    for i in range(numIter):
        # print centers
        respMatrix = eStep(data, beta, centers)
        centers = mStep(data, respMatrix, m)
        # print respMatrix
        
    return centers

# k, m, beta, data = readMatrixFile('soft_kMeansMatrix.txt', betaParam=True)
# centers = soft_kMeans(data, k, m, beta, numIter=100)
# for center in centers:
    # print ' '.join([ str(x) for x in center ])
    
    
