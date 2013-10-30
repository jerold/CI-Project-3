#!/usr/bin/python

import json
import itertools

import math
import random

pageBlock = {'inFiles':['data/block/page-blocks.data'],
             'outFile':'data/block/pageblocks.json',
             'width':10,
             'height':1}

car = {'inFiles':['data/car/car.data'],
             'outFile':'data/car/car.json',
             'width':10,
             'height':1}

flare = {'inFiles':['data/flare/flare.data1',
                    'data/flare/flare.data2'],
             'outFile':'data/flare/flare.json',
             'width':10,
             'height':1}

flare = {'inFiles':['data/houseing/houseing.data'],
             'outFile':'data/flare/flare.json',
             'width':10,
             'height':1}


def parseFlare(lines):
    patSet = []
    attributes = [[] for _ in range(10)]
    attributesAlt = [{'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'H':7},
                     {'X':1, 'R':2, 'S':3, 'A':4, 'H':5, 'K':6},
                     {'X':1, 'O':2, 'I':3, 'C':4}]
    targets = [[] for _ in range(3)]
    for line in lines:
        line = line.split('\n')[0]
        line = line.split()
        pattern = line[:-3]
        for i, elem in enumerate(pattern):
            if i < 3:
                pattern[i] = attributesAlt[i][elem]
            else:
                pattern[i] = int(elem)
            attributes[i].append(pattern[i])
        patternTarget = line[-3:]
        patSet.append({'p':pattern, 't':patternTarget})
        targets[0].append(patternTarget[0])
        targets[1].append(patternTarget[1])
        targets[2].append(patternTarget[2])
        #print('p:' + str(pattern) + '  t:' + str(patternTarget))
    print(set(targets[0]))
    print(set(targets[1]))
    print(set(targets[2]))
    print("Attributes")
    for attribute in attributes:
        print(set(attribute))
    return patSet

def parseCar(lines):
    patSet = []
    attributes = [[] for _ in range(6)]
    attributesAlt = [{'low':1, 'vhigh':4, 'med':2, 'high':3},
                     {'low':1, 'vhigh':4, 'med':2, 'high':3},
                     {'4':3, '5more':4, '2':1, '3':2},
                     {'4':2, 'more':3, '2':1},
                     {'big':3, 'med':2, 'small':1},
                     {'med':2, 'low':1, 'high':3}]
    targets = []
    targetsAlt = {'acc':2, 'good':3, 'unacc':1, 'vgood':4}
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        pattern = line[:len(line)-1]
        for i, elem in enumerate(pattern):
            pattern[i] = attributesAlt[i][elem]
            attributes[i].append(pattern[i])
        patternTarget = targetsAlt[line[len(line)-1]]
        patSet.append({'p':pattern, 't':patternTarget})
        targets.append(patternTarget)
        #print('p:' + str(pattern) + '  t:' + str(patternTarget))
    print(set(targets))
    print("Attributes")
    for attribute in attributes:
        print(set(attribute))
    return patSet

def parseBlock(lines):
    patSet = []
    targets = []
    for line in lines:
        line = line.split('\n')[0]
        line = line.split()
        pattern = line[:len(line)-1]
        for i in range(len(pattern)):
            pattern[i] = float(pattern[i])
        patternTarget = int(line[len(line)-1])
        patSet.append({'p':pattern, 't':patternTarget})
        targets.append(patternTarget)
        #print('p:' + str(pattern) + '  t:' + str(patternTarget))
    print(set(targets))
    return patSet


def mygrouper(n, iterable):
    "http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n"
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.izip_longest(*args))

def buildKMeansCenters(patterns, w, h, k):
    centers = {}
    for i in range(k):
        centers[i] = emptyPattern(w, h)
        if h > 1:
            centers[i][random.randint(0,h-1)][random.randint(0,w-1)] = 1
        else:
            centers[i][random.randint(0,w-1)] = 1
    dist = 100
    while dist > 3:
        tempCenters = adjustCenters(patterns, centers)
        dist = 0
        for i in range(k):
            dist = dist + euclidianDistance(centers[i], tempCenters[i])
        centers = tempCenters
        print(dist)
    for i in range(k):
        printPattern(centers[i])
        print(i)
    return centers

def adjustCenters(patterns, centers):
    groups = {}
    for k in centers.keys():
        groups[k] = []
    for pattern in patterns:
        bestDist = 99999
        bestKey = ''
        for key in centers.keys():
            center = centers[key]
            dist = euclidianDistance(pattern['p'], center)
            if dist < bestDist:
                bestDist = dist
                bestKey = key
        groups[bestKey].append(pattern)
    newCenters = {}
    for k in centers.keys():
        if len(groups[k]) > 0:
            newCenters[k] = buildMeanPattern(groups[k])
        else:
            newCenters[k] = centers[k]
    return newCenters

def euclidianDistance(p, q):
    sumOfSquares = 0.0
    if isinstance(p[0], list):
        for i in range(len(p)):
            for j in range(len(p[i])):
                sumOfSquares = sumOfSquares + ((p[i][j]-q[i][j])*(p[i][j]-q[i][j]))
    else:
        for i in range(len(p)):
            sumOfSquares = sumOfSquares + ((p[i]-q[i])*(p[i]-q[i]))
    return math.sqrt(sumOfSquares)


def buildCentersAndSigmas(patterns):
    centersTargets = {}
    for pattern in patterns:
        if pattern['t'] not in centersTargets:
            centersTargets[pattern['t']] = []
        centersTargets[pattern['t']].append(pattern)
    centers = {}
    sigmas = {}
    print("Found " + str(len(centersTargets)) + " targets.")
    # build center as mean of all trained k patterns, and sigma as standard deviation
    for k in centersTargets.keys():
        kPats = centersTargets[k]
        centers[k] = buildMeanPattern(kPats)

    print("Centers Post Average:")
    for k in centersTargets.keys():
        print(k)
        printPattern(centers[k])

    # Build Sigmas for each space
    print("Sigma:")
    for k in centersTargets.keys():
        sigmas[k] = buildSigmaPattern(centers[k], kPats)
        printPattern(sigmas[k])

    # refine centers using k-means
    dist = 100
    distDelta = 100
    oldDist = 0
    while dist > 1 and abs(distDelta) > 0.01:
        tempCenters = adjustCenters(patterns, centers)
        dist = 0
        for k in centersTargets.keys():
            dist = dist + euclidianDistance(centers[k], tempCenters[k])
        centers = tempCenters
        distDelta = dist - oldDist
        oldDist = dist
        print("dist:" + str(dist) + ", delta:" + str(distDelta))

    print("Centers Post K-means:")
    for k in centersTargets.keys():
        print(k)
        printPattern(centers[k])

    return {'centers':centers, 'sigmas':sigmas}

def buildMeanPattern(patterns):
    h = 0
    w = len(patterns[0]['p'])
    if isinstance(patterns[0]['p'][0], list):
        h = len(patterns[0]['p'])
        w = len(patterns[0]['p'][0])
    mPat = emptyPattern(w, h)
    for pat in patterns:
        #print(pat['p'])
        if h > 1:
            #print(str(len(pat['p'])) + " x " + str(len(pat['p'][0])))
            for i in range(h):
                for j in range(w):
                    mPat[i][j] = mPat[i][j] + pat['p'][i][j]
        else:
            for j in range(w):
                mPat[j] = mPat[j] + pat['p'][j]
    if h > 1:
        for i in range(h):
            for j in range(w):
                mPat[i][j] = mPat[i][j] / len(patterns)
    else:
        for j in range(w):
            mPat[j] = mPat[j] / len(patterns)
    return mPat

def buildSigmaPattern(meanPat, patterns):
    h = 0
    w = len(patterns[0]['p'])
    if isinstance(patterns[0]['p'][0], list):
        h = len(patterns[0]['p'])
        w = len(patterns[0]['p'][0])
    sPat = emptyPattern(w, h)
    # Sum over all square of distance from means
    if h > 1:
        for i in range(h):
            for j in range(w):
                for pat in patterns:
                    sPat[i][j] = sPat[i][j] + (pat['p'][i][j] - meanPat[i][j])*(pat['p'][i][j] - meanPat[i][j])
                sPat[i][j] = math.sqrt(1.0/len(patterns)*sPat[i][j])
    else:
        for j in range(w):
            for pat in patterns:
                sPat[j] = sPat[j] + (pat['p'][j] - meanPat[j])*(pat['p'][j] - meanPat[j])
            sPat[j] = math.sqrt(1.0/len(patterns)*sPat[j])
    return sPat
        


def emptyPattern(w, h):
    pat = []
    if h > 1:
        for i in range(h):
            pat.append([])
            for j in range(w):
                pat[i].append(0.0)
    else:
        for j in range(w):
            pat.append(0.0)
    return pat

def printPattern(pattern):
    tolerance = 0.7
    if isinstance(pattern[0], list):
        for i in range(len(pattern)):
            print(', '.join(str(round(x+tolerance,3)) for x in pattern[i]))
    else:
        print(', '.join(str(round(x,3)) for x in pattern))


if __name__=="__main__":
    #parseSet = pageBlock
    #parseSet = car
    parseSet = flare
    
    lines = []
    for fileName in parseSet['inFiles']:
        with open(fileName) as file:
            fileLines = file.readlines()
            for line in fileLines:
                lines.append(line)
                
    #patternSet = parseBlock(lines)
    #patternSet = parseCar(lines)
    patternSet = parseFlare(lines)
       
    print("pats: " + str(len(patternSet)))
    with open(parseSet['outFile'], 'w+') as outfile:
        data = {'count':len(patternSet),
                'width':parseSet['width'],
                'height':parseSet['height'],
                'patterns':patternSet}
        json.dump(data, outfile)


