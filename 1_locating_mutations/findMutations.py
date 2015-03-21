#!/usr/bin/env python

import os, re, sys


# Construct the trie data structure for the passed set of patterns
def trieConstruction(patterns):
    trie = dict()
    
    for pattern in patterns:
        currentNode = trie
        
        for i in range(len(pattern)):
            if pattern[i] in currentNode:
                currentNode = currentNode[ pattern[i] ]
            else:
                newNode = dict()
                currentNode[ pattern[i] ] = newNode
                currentNode = newNode
                
    return trie
    
    
# Print an adjacency list for the nodes in the passed trie
def printTrieAdjacency(trie, num=0, total=0):
    for node in sorted(trie):
        print '%d->%d:%s' % (num, total+1, node)
        total = printTrieAdjacency(trie[node], max(total+1, num+1), total+1)
        
    return total
        
# trie = trieConstruction(['ATAGA','ATC','GAT'])
# printTrieAdjacency(trie)


# If the the passed trie conatins a match in the passed text, return 
# the portion of text that matches or None if there is no match
def prefixTrieMatching(text, trie):
    pos = 0
    v = trie
    
    while pos < len(text):
        if len(v)==0:
            return text[:pos]
        elif text[pos] in v:
            v = v[ text[pos] ]
            pos += 1
        else:
            return None

# looks for a match of the passed trie at all positions in the
# passed text.  Returns all positions where there is  match. 
def trieMatching(text, trie):
    positions = []
    for i in range(len(text)):
        if prefixTrieMatching(text[i:], trie) is not None:
            positions.append(i)
            
    return positions

# trie = trieConstruction(['ATCG','GGGT'])
# print trieMatching('AATCGGGTTCAATCGGGGT', trie)


# class Edge:
    # def __init__(self, symbol, pos):
        # self.symbol = symbol
        # self.pos = pos
    # def __hash__(self):
        # return hash(self.symbol)
    # def __eq__(self, other):
        # return self.symbol==other.symbol and self.pos==other.pos
    # def updatePos(self, pos):
        # if pos < self.pos:
            # self.pos=pos
    
    
    
    
# def nodeContainsChar(currentNode, char):
    # for charTuple in currentNode:
        # if charTuple[0]==char:
            # return True
    # return False
    
    
# Construct a suffix trie
def modifiedSuffixTrieConstruction(seq):
    trie = dict()
    for i in range(len(seq)):
        currentNode = trie
        
        for j in range(i, len(seq)):
            if seq[j] not in currentNode:
                newNode=i if j==len(seq)-1 else dict()
                currentNode[ seq[j] ] = (j, newNode)
            if j < len(seq)-1:
                currentNode = currentNode[ seq[j] ][1]
            else:
                
    return trie
    
    
trie = modifiedSuffixTrieConstruction('ATAAATG$')
print trie
                