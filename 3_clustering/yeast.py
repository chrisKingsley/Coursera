#!/usr/bin/env python

import re, sys


def readMatrixFile(fileName):
    matrix = []
    infile = open(fileName, 'r')
    k, m = [ int(x) for x in infile.readline().rstrip().split() ]
    for line in infile:
        matrix.append([ float(x) for x in line.rstrip().split() ]

def farthestFirstTraversal(k, matrix):
    

k, m, matrix = readMatrixFile('geMatrix.txt')
