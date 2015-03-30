#!/usr/bin/env python

import os, re, sys
from collections import Counter


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
                
    return trie


def getTrieEdges(trie, edge='', edges=[]):
    for char, node in trie.iteritems():
        #print char, node
        if isinstance(node[1], dict):
            if len(node[1])>1:
                edges.append( edge + char )
                getTrieEdges(node[1], '')
            else:
                getTrieEdges(node[1], edge + char)
        else:
            edges.append( edge + char )
        
    return edges

# trie = modifiedSuffixTrieConstruction('ATAAATG$')
# print getTrieEdges(trie)
    
    
def getLongestRepeat(trie, edge='', longRep='', depth=0):
    while len(trie)==1:
        char, node = trie.popitem()
        edge += char
        if isinstance(node[1], dict):
            trie = node[1]
    
    if len(trie) > 1:
        if len(edge) > len(longRep):
            longRep = edge
        for char, node in trie.iteritems():
            if isinstance(node[1], dict):
                longRep = getLongestRepeat(node[1], edge+char, longRep, depth+1)

    return longRep

# trie = modifiedSuffixTrieConstruction('ATATCGTTTTATCGTT$')
# print trie
# print getLongestRepeat(trie)  # should be TATCGTT


def seqInTrie(trie, seq):
    #print seq
    for i in range(len(seq)):
        
        if isinstance(trie, dict) and seq[i] in trie:
            node = trie[ seq[i] ]
            trie = node[1]
            #print seq[i], "True"
        else:
            #print seq[i], "False"
            return False
    return True
    
    
def getLongestSharedSubstring(trie, seq):
    longSeq = ''
    
    for i in range(len(seq)):
        for j in range(i+1, len(seq)+1):
            if (j-i) > len(longSeq) and seqInTrie(trie, seq[i:j]):
                longSeq = seq[i:j]
                print (j-i), seq[i:j]
                
    return longSeq
    
# seq1, seq2 = 'TCGGTAGATTGCGCCCACTC','AGGGGCTCGCAGTGTAAGAA'
# trie = modifiedSuffixTrieConstruction(seq2)
# print trie.keys()
# print getLongestSharedSubstring(trie, seq1)
    
    
def getShortestUnsharedSubstring(trie, seq):
    shortSeq = seq
    
    for i in range(len(seq)):
        for j in range(i+1, len(seq)+1):
            if (j-i) < len(shortSeq) and not seqInTrie(trie, seq[i:j]):
                shortSeq = seq[i:j]
                print (j-i), seq[i:j]
                
    return shortSeq
    
# seq1, seq2 = 'CCAAGCTGCTAGAGG', 'CATGCTGGGCTGGCT'
# trie = modifiedSuffixTrieConstruction(seq2)
# print trie.keys()
# print getShortestUnsharedSubstring(trie, seq1) # should be CC


def getSuffixArray(seq):
    suffixArray = []
    for i in range(len(seq)):
        suffixArray.append(seq[i:])
    suffixArray.sort()
    
    for i in range(len(suffixArray)):
        suffixArray[i] = len(seq)-len(suffixArray[i])
        
    return suffixArray
    
# suffixArray = getSuffixArray('AACGATAGCGGTAGA$')
# print ', '.join([ str(x) for x in suffixArray ])


