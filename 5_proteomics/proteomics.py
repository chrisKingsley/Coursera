#!/usr/bin/env python

import re, sys

def loadSpectrum(file):
    infile = open(file, 'r')
    spectrum = [ int(x) for x in infile.readline().split() ]
    infile.close()
    
    return [0] + spectrum
    

def loadAA_masses(file):
    aaTable = dict()
    
    infile = open(file, 'r')
    for line in infile:
        aa, mass = line.rstrip().split()
        aaTable[ int(mass) ] = aa
    infile.close()
    
    return aaTable
    
    
def spectrumGraph(spectrum, aaTable):
    for i in range(len(spectrum)-1):
        for j in range(i+1, len(spectrum)):
            mass = spectrum[j] - spectrum[i]
            print i, j, mass

aaTable = loadAA_masses('AA_masses.txt')
spectrum = loadSpectrum('spectrum.txt')
print spectrumGraph(spectrum, aaTable)

