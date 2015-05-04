#!/usr/bin/env python

import re

leafSeqs = set()

probFile = open('C:/My_Stuff/prob.txt', 'r')
for line in probFile:
    m = re.search(r'\d+\-\>([ACGT]+)', line)
    if m:
        leafSeqs.add(m.group(1))
probFile.close()
print '%d leaf seqs read' % len(leafSeqs)


mySolnFile = open('C:/My_Stuff/mySoln.txt', 'r')
numMatches1, numMatches2 = 0, 0
for line in mySolnFile:
    seq1, seq2 = re.split('->|\:', line)[0:2]
    if seq1 in leafSeqs:
        numMatches1 += 1
    if seq2 in leafSeqs:
        numMatches2 += 1
mySolnFile.close()

print 'num matches1 %d numMatches2 %d' % (numMatches1, numMatches2)