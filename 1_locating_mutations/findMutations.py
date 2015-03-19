#!/usr/bin/env python

import os, re, sys

# construct the trie data structure for the passed set of patterns
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
    
    
def printTrieAdjacency(trie, num=0, total=0):
    for node in sorted(trie):
        print '%d->%d:%s' % (num, total+1, node)
        total = printTrieAdjacency(trie[node], max(total+1, num+1), total+1)
        
    return total
        
# trie = trieConstruction(['ATAGA','ATC','GAT'])
# printTrieAdjacency(trie)


def prefixTrieMatching(text, trie):
    pos = 0
    v = trie
    
    while pos < len(text):
        if len(v)==0:
            return text[:pos]
        elif text[pos] in v:
            #print text[pos], pos, len(text), text
            v = v[ text[pos] ]
            pos += 1
            # if pos==len(text):
                # return text
        else:
            #print 'no matches found'
            return None
            
def trieMatching(text, trie):
    positions = []
    for i in range(len(text)):
        if prefixTrieMatching(text[i:], trie) is not None:
            positions.append(i)
            
    return positions

# trie = trieConstruction(['ATCG','GGGT'])
# print trieMatching('AATCGGGTTCAATCGGGGT', trie)


suffixTrieConstruction(seq):
    trie = dict()
    for i in range(len(seq)):
        currentNode = trie
        for j in range(i, len(seq)):
            if seq[j] in currentNode:
                currentNode = currentNode[ seq[j] ]
            else
                newNode = dict()
                currentNode[ seq[j] ] = (seq[j], j)