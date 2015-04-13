#!/usr/bin/env python

import math, os, random, re, sys
from collections import Counter

BASE_DICT = {'A':0, 'C':1, 'G':2, 'T':3, '$':4}


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


# return the sequence labels for the edges of the passed trie
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
    
    
# return the longest repeated sequence in the passed trie
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


# returns whether the passed sequence is present in the passed trie
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
    
    
# returns the longest shared substring of the passed sequence and trie
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
    
    
# returns the shortest unshared substring of the passed sequence and trie
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


# returns the suffix array of the passed sequence
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


# performs the Burroughs-Wheeler transformation of the passed sequence
def BW_transform(seq):
    seqArray = ['']*len(seq)
    retVal = ''
    
    for i in range(len(seq)):
        seqArray[i] = seq[i:] + seq[:i]
    seqArray.sort()
    
    for i in range(len(seqArray)):
        retVal += seqArray[i][len(seq)-1]
        
    return retVal
    
# print BW_transform('GCGTGCCTGGTCA$')


# gets the count of a character in the passed dict and increments it
def getCharCount(char, charDict):
    count = charDict.get(char, 0) + 1
    charDict[ char ] = count
    
    return count

    
# performs the Burroughs-Wheeler inverse transformation of the passed sequence
def BW_inverse(seq, endchar='$'):
    stringPattern = '%%s%%0%dd' % int(math.log10(len(seq))+1)
    charCounts1, charCounts2, charDict = dict(), dict(), dict()
    
    col1 = ''.join(sorted(seq))
    for i in range(len(seq)):
        count1 = getCharCount(col1[i], charCounts1)
        count2 = getCharCount(seq[i], charCounts2)
        key = stringPattern % (seq[i], count2)
        val = stringPattern % (col1[i], count1)
        charDict[ key ] = val
        
    retVal, startString = '', stringPattern % (endChar, 1)
    char = startString
    while True:
        char = charDict[char]
        retVal += re.sub('\d', '', char)
        if char==startString:
            break
            
    return retVal

# print BW_inverse('enwvpeoseu$llt')
# print BW_inverse('TTCCTAACG$A')


# returns the lastToFirst array that contains pointers from character
# positions in the last column of the BW matrix to the first column
def getLastToFirstArray(firstCol, lastCol):
    charPos, charCounts1, charCounts2 = dict(), dict(), dict()
    lastToFirst = [0] * len(lastCol)
    stringPattern = '%%s%%0%dd' % int(math.log10(len(firstCol))+1)
    
    for i in range(len(firstCol)):
        count = getCharCount(firstCol[i], charCounts1)
        key = stringPattern % (firstCol[i], count)
        charPos[ key ] = i
        
    for i in range(len(lastCol)):
        count = getCharCount(lastCol[i], charCounts2)
        key = stringPattern % (lastCol[i], count)
        lastToFirst[i] = charPos[ key ]
        
    return lastToFirst
    
    
# returns the number of matches of the pattern in the sequence
# defined by the lastColumn of the BW matrix
def BW_match(firstCol, lastCol, pattern, lastToFirst):
    top, bottom, pos = 0, len(lastCol)-1, len(pattern)-1
    
    while top <= bottom:
        topIndex, bottomIndex = sys.maxint, -1
        for i in range(top, bottom+1):
            if pattern[pos]==lastCol[i]:
                if i < topIndex:
                    topIndex = i
                if i > bottomIndex:
                    bottomIndex = i
                    
        if topIndex<sys.maxint and bottomIndex>-1:
            top = lastToFirst[topIndex]
            bottom = lastToFirst[bottomIndex]
            print pattern[pos], top, bottom
        else:
            print pattern, "not found"
            return 0
                
        pos -= 1
        if pos < 0:
            return bottom - top + 1
                
            

# performs BW matching on each pattern in patterns against the
# BW inverse transform of the passed character sequence
def BW_Matching(seq, patterns):
    lastCol = seq
    firstCol = ''.join(sorted(seq))
    lastToFirst = getLastToFirstArray(firstCol, lastCol)
    retVal = []
    
    for pattern in patterns:
        numMatch = BW_match(firstCol, lastCol, pattern, lastToFirst)
        retVal.append( numMatch )
        
    return retVal
        
# seq = 'TCCTCTATGAGATCCTATTCTATGAAACCTTCA$GACCAAAATTCTCCGGC'
# patterns = ['CCT','CAC','GAG','CAG','ATC']
# print BW_Matching('TCCTCTATGAGATCCTATTCTATGAAACCTTCA$GACCAAAATTCTCCGGC', patterns)


# returns a suffix dictionary for every k position in the sequence
def getPartialSuffixDict(seq, k=5):
     suffArray = getSuffixArray(seq)
     retVal = dict()
     
     for i in range(len(suffArray)):
         if suffArray[i] % k == 0:
             retVal[ i ] = suffArray[i]
             
     return retVal
     
#print getPartialSuffixDict('PANAMABANANAS$', 5)


# Return dict containing the first position of each character in the
# first column of the BW matrix
def getFirstOccurrenceDict(seq):
    retVal = {}
    for i in range(len(seq)):
        retVal[ seq[i] ] = retVal.get(seq[i], 0) + 1
    
    start = 0
    for char in sorted(retVal):
        retVal[ char ], start  = start, start + retVal[ char ]
        
    return retVal


# returns the count matrix for all characters in BASE_DICT in 
# the passed sequence
def getCountMatrix(seq, k=5):
    counts = [0]*len(BASE_DICT)
    retVal = [counts[:]]
    
    for i in range(len(seq)):
        idx = BASE_DICT[ seq[i] ]
        counts[ idx ] += 1
        if i % k==0:
            retVal.append(counts[:])
            
    return retVal

    
# returns the count dict for all characters in BASE_DICT in 
# the passed sequence at every k positions
def getCountDict(seq, k=5):
    counts = [0]*len(BASE_DICT)
    retVal = { 0:counts[:] }

    for i in range(1, len(seq)+1):
        idx = BASE_DICT[ seq[i-1] ]
        counts[ idx ] += 1
        if i % k==0:
            retVal[i] = counts[:]
            
    return retVal


# returns the base count at the specified position using the
# passed count dictionary    
def getBaseCount(lastCol, count, char, pos):
    numMatches = 0
    
    while pos not in count:
        if lastCol[pos-1]==char:
            numMatches += 1
        pos -= 1
    
    idx = BASE_DICT[char]
    return count[pos][idx] + numMatches
        
        
# get the starting positions of the matches in the original
# sequence given the top/bottom positions of the final match
def getStartPositions(lastCol, firstOccurrence, count,
                      suffixDict, top, bottom):
    retVal = []
    
    for i in range(top, bottom+1):
        pos, numSteps = i, 0
        while pos not in suffixDict:
            char = lastCol[pos]
            idx = BASE_DICT[char]
            baseCount = getBaseCount(lastCol, count, char, pos)
            pos = firstOccurrence[char] + baseCount
            numSteps += 1
        retVal.append(suffixDict[pos] + numSteps)

    return retVal
    
    
# Burrough's Wheeler matching using count matrices
def BW_match2(lastCol, pattern, firstOccurrence, count, suffixDict):
    top, bottom, pos = 0, len(lastCol)-1, len(pattern)-1
    
    while top <= bottom:
        char = pattern[pos]
        idx = BASE_DICT[char]
        topCount = getBaseCount(lastCol, count, char, top)
        bottomCount = getBaseCount(lastCol, count, char, bottom+1)
        if topCount!=bottomCount:
            top = firstOccurrence[char] + topCount
            bottom = firstOccurrence[char] + bottomCount - 1
        else:
            return []
                
        pos -= 1
        if pos < 0:
            return getStartPositions(lastCol, firstOccurrence, 
                        count, suffixDict, top, bottom)
    
    
# performs less memory intensive BW matching on each pattern in
# patterns against the BW inverse transform of the passed character
# sequence
def BW_Matching2(seq, patterns, k=5):
    lastCol = BW_transform(seq)
    firstOccurrence = getFirstOccurrenceDict(lastCol)
    suffixDict = getPartialSuffixDict(seq, k=k)
    count = getCountDict(lastCol, k=k)
    retVal = []
    
    for pattern in patterns:
        matchPos = BW_match2(lastCol, pattern, firstOccurrence, count, suffixDict)
        retVal += matchPos
        
    return sorted(retVal)
    
# seq = 'AATCGGGTTCAATCGGGGT$'
# patterns = ['ATCG','GGGT']
# results = BW_Matching2(seq, patterns, k=5)


# gets the seeds for the passed pattern given the allowed number of
# mismatches d
def getPatternSeeds(pattern, d):
    matchLength = len(pattern)/(d+1)
    start = 0
    retVal = []
    
    for i in range(d):
        retVal.append(pattern[start:(start+matchLength)])
        start += matchLength
    retVal.append(pattern[start:])
        
    return retVal


# get the position of the match of the pattern, given the
# list of seeds and the index of the seed that matched
def getApproxMatchPosition(seeds, matchPos, seed_idx):
    for i in range(seed_idx):
        matchPos -= len(seeds[i])
    return matchPos 


# return the number of mismatches between pattern and seq at
# the passed position in seq
def getNumMismatches(seq, pattern, pos):
    numMismatch = 0
    
    for i in range(len(pattern)):
        if pos+i<0 or pos+i>=len(seq) or seq[pos+i]!=pattern[i]:
            numMismatch += 1
            
    return numMismatch


# returns the match positions of the passed pattern in the passed sequence given
# the number of allowed mismatches d
def approximateMatch(seq, pattern, d, lastCol, firstOccurrence, count, suffixDict):
    seeds = getPatternSeeds(pattern, d)
    matchPosSet = set()
    
    for i in range(len(seeds)):
        seedMatches = BW_match2(lastCol, seeds[i], firstOccurrence, count, suffixDict)
        if len(seedMatches) > 0:
            for seedMatchPos in seedMatches:
                matchPos = getApproxMatchPosition(seeds, seedMatchPos, i)
                numMismatch = getNumMismatches(seq, pattern, matchPos)
                if numMismatch <= d:
                    matchPosSet.add(matchPos)
                    
    return [ x for x in matchPosSet ]
    
    
# returns the match positions of the passed patterns in the passed sequence given
# the number of allowed mismatches d
def approximateMatching(seq, patterns, d, k=5):
    lastCol = BW_transform(seq)
    firstOccurrence = getFirstOccurrenceDict(lastCol)
    suffixDict = getPartialSuffixDict(seq, k=k)
    count = getCountDict(lastCol, k=k)
    retVal = []
    
    for pattern in patterns:
        retVal += approximateMatch(seq, pattern, d, lastCol, firstOccurrence, count, suffixDict)
        
    return retVal
    

# seq = 'ACATGCTACTTT$'
# patterns = ['ATT','GCC','GCTA','TATT']
# print sorted(approximateMatching(seq, patterns, 1))



# QUIZ
# trie = modifiedSuffixTrieConstruction('TCTGAGCCCTACTGTCGAGAAATATGTATCTCGCCCCCGCAGCTT$')
# edges = [ x for x in getTrieEdges(trie) if x.endswith('$') ]
# print len(edges), edges

# print BW_transform('GATTGCTTTT$')
# print BW_inverse('AT$AAACTTCG')
