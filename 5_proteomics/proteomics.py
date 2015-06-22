#!/usr/bin/env python

import re, sys


# load a space delimited file of spectral masses
def loadSpectrum(file):
    infile = open(file, 'r')
    spectrum = [ int(x) for x in infile.readline().split() ]
    infile.close()
    
    return [0] + spectrum
    

# load mass to amino acid (and vice versa) values
# into a dictionary
def loadAA_masses(file):
    massToAA, aaToMass = dict(), dict()
    
    infile = open(file, 'r')
    for line in infile:
        aa, mass = line.rstrip().split()
        massToAA[ int(mass) ] = aa
        aaToMass[ aa ] = int(mass)
    infile.close()
    
    return massToAA, aaToMass
    
massToAA, aaToMass = loadAA_masses('AA_masses.txt')

    
# returns the mass of the passed peptide sequence
def peptideMass(peptide, aaToMass):
    mass = 0
    for i in range(len(peptide)):
        mass += aaToMass[ peptide[i] ]
    
    return mass
    
    
def spectrumGraph(spectrum, massToAA, printGraph=False):
    graph = dict()
    
    for i in range(len(spectrum)-1):
        for j in range(i+1, len(spectrum)):
            mass = spectrum[j] - spectrum[i]
            if mass in massToAA:
                if spectrum[i] not in graph:
                    graph[ spectrum[i] ] = []
                graph[ spectrum[i] ].append(spectrum[j])
                if printGraph:
                    print '%s->%s:%s' % \
                        (spectrum[i], spectrum[j], massToAA[mass])
    
    return graph

# spectrum = loadSpectrum('spectrum.txt')
# specGraph = spectrumGraph(spectrum, massToAA, True)


def idealSpectrum(peptide, aaToMass):
    spectrum = set()
    for i in range(len(peptide)):
        spectrum.add(peptideMass(peptide[:i], aaToMass))
        spectrum.add(peptideMass(peptide[i:], aaToMass))
    
    return sorted(spectrum)
    
    
def spectraEquivalent(peptide, spectrum, aaToMass):
    idealSpec = idealSpectrum(peptide, aaToMass)
    return idealSpec==spectrum

    
def decodeIdealSpecturm(spectrum, graph, aaToMass, massToAA, peptide='', mass=0):
    if mass not in graph:
       return
    
    for mass2 in graph[ mass ]:
        newPeptide = peptide + massToAA[ mass2 - mass ]
        if spectraEquivalent(newPeptide, spectrum, aaToMass):
            print newPeptide
        else:
            decodeIdealSpecturm(spectrum, graph, aaToMass, massToAA, newPeptide, mass2)

# spectrum = loadSpectrum('spectrum.txt')
# graph = spectrumGraph(spectrum, massToAA)
# decodeIdealSpecturm(spectrum, graph, aaToMass, massToAA)

def peptideToVector(peptide, aaToMass):
    pepVec = [0]*peptideMass(peptide, aaToMass)
    idx = 0
    for i in range(len(peptide)):
        idx += aaToMass[ peptide[i] ]
        pepVec[ idx-1 ] = 1
        
    return pepVec
        
# print peptideToVector('XZZXX', {'X':4, 'Z':5})
# pepVec = peptideToVector('DNMLHRALKPVFQQKPYPSYVWWEKGLR', aaToMass)
# print ' '.join([ str(x) for x in pepVec ])


def vectorToPeptide(pepVec, massToAA):
    peptide, mass = '', 0
    for i in range(len(pepVec)):
        if int(pepVec[i])==1:
            peptide  += massToAA[ i-mass+1 ]
            mass = i+1
            
    return peptide
    
# pepVec = '0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1'.split()
# print vectorToPeptide(pepVec, {4:'X', 5:'Z'})

