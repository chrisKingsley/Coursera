#!/usr/bin/env python

import os, re, sys

# construct the trie data structure for the passed set of patterns
def trieConstruction(patterns):
    trie = dict()
    
    for pattern in patterns:
        print pattern
        
        
trieConstruction(['ATAGA','ATC','GAT'])