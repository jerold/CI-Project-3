#!/usr/bin/python
# patternSet.py

import json
import random
import math

def findUniqueTargets(patterns):
    targets = []
    counts = {}
    for pattern in patterns:
        if pattern['t'] not in targets:
            targets.append(pattern['t'])
            counts[pattern['t']] = 1
        else:
            counts[pattern['t']] = counts[pattern['t']] + 1
    targets.sort()
    print("Targets: [" + ", ".join(str(t) + "x" + str(counts[t]) for t in targets) + "]")
    return {'targets':targets, 'counts':counts}

# Creates and empty pattern of the given dimensionality
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

# print an individual pattern with or without target value
def printPatterns(pattern):
    if isinstance(pattern, dict):
        for key in pattern.keys():
            if key == 't':
                print("Target: " + str(key))
            elif key == 'p':
                printPatterns(pattern['p'])
    elif isinstance(pattern[0], list):
        for pat in pattern:
            printPatterns(pat)
    else:
        print(', '.join(str(round(x, 3)) for x in pattern))

# A Pattern set contains sets of 3 types of patterns
# and can be used to retrieve only those patterns of a certain type
class PatternSet:
    # Reads patterns in from a file, and puts them in their coorisponding set
    def __init__(self, fileName, percentTraining):
        with open(fileName) as jsonData:
            data = json.load(jsonData)
            
        # Assign Patterns and Randomize order
        self.patterns = data['patterns']
        self.count = data['count']
        self.inputMagX = len(self.patterns[0]['p'])
        self.inputMagY = 1
        if isinstance(self.patterns[0]['p'][0], list):
            self.inputMagX = len(self.patterns[0]['p'][0])
            self.inputMagY = len(self.patterns[0]['p'])

        targetsWithCounts = findUniqueTargets(self.patterns)
        self.targets = targetsWithCounts['targets']
        self.counts = targetsWithCounts['counts']

        # Use this if we wish each category to have the same number of patterns for it
        keys = self.counts.keys()
        minPatternCount = 9999
        maxPatternCount = 0
        patternTargetSets = {}
        self.newPatterns = []
        for key in keys:
            if minPatternCount > self.counts[key]:
                minPatternCount = self.counts[key]
            if maxPatternCount < self.counts[key]:
                maxPatternCount = self.counts[key]
        for key in keys:
            patternTargetSets[key] = []
            for pat in self.patterns:
                if pat['t'] == key:
                    patternTargetSets[key].append(pat)
            setMultiplier = 0
            while setMultiplier*len(patternTargetSets[key]) < maxPatternCount:
                if (setMultiplier+1)*len(patternTargetSets[key]) > maxPatternCount:
                    addAmount = (setMultiplier+1)*len(patternTargetSets[key]) - maxPatternCount
                    self.newPatterns = self.newPatterns + patternTargetSets[key][:addAmount]
                else:
                    self.newPatterns = self.newPatterns + patternTargetSets[key]
                setMultiplier = setMultiplier + 1
        self.patterns = self.newPatterns
        self.count = len(self.patterns)

        targetsWithCounts = findUniqueTargets(self.patterns)
        self.targets = targetsWithCounts['targets']
        self.counts = targetsWithCounts['counts']

        random.shuffle(self.patterns)
        print(str(len(self.patterns)) + " Patterns Available (" + str(self.inputMagY) + "x" + str(self.inputMagX) + ")")

        # Architecture has 1 output node for each digit / letter
        # Assemble our target and confusion matrix
        keys = self.targets
        keys.sort()
        self.confusionMatrix = {}
        self.targetMatrix = {}
        index = 0
        for key in keys:
            self.confusionMatrix[key] = [0.0] * len(keys)
            self.targetMatrix[key] = [0] * len(keys)
            self.targetMatrix[key][index] = 1
            index = index + 1
        self.outputMag = len(keys)

    def printConfusionMatrix(self):
        keys = list(self.confusionMatrix.keys())
        keys.sort()
        print("\nConfusion Matrix")
        for key in keys:
            printPatterns(self.confusionMatrix[key])
        print("\nKey, Precision, Recall")
        #for key in keys:
            #print(str(key) + ", " + str(round(self.calcPrecision(key), 3)) + ", " + str(round(self.calcRecall(key), 3)))
        self.calcPrecisionAndRecall()

    def calcPrecision(self, k):
        tp = self.confusionMatrix[k][k]
        fpSum = sum(self.confusionMatrix[k])
        if fpSum == 0.0:
            return fpSum
        return tp/fpSum

    def calcRecall(self, k):
        tp = self.confusionMatrix[k][k]
        keys = list(self.confusionMatrix.keys())
        keys.sort()
        i = 0
        tnSum = 0.0
        for key in keys:
            tnSum = tnSum + self.confusionMatrix[key][k]
        if tnSum == 0.0:
            return tnSum
        return tp/tnSum

    def calcPrecisionAndRecall(self):
        keys = list(self.confusionMatrix.keys())
        matrixSum = 0.0
        keys.sort()
        i = 0
        precision = []
        recall = []
        diagonal = []
        for key in keys:
            row = self.confusionMatrix[key]
            rowSum = 0
            for j, val in enumerate(row):
                if i==j:
                    diagonal.append(val)
                rowSum += val
                if len(recall) == j:
                    recall.append(val)
                else:
                    recall[j] = recall[j] + val
            matrixSum = matrixSum + rowSum
            precision.append(rowSum)
            i += 1
        for i, elem in enumerate(diagonal):
            if abs(precision[i]) > 0.0 and abs(recall[i]) > 0.0:
                print(str(keys[i]) + ", " + str(round(elem / precision[i], 4)) + ", " + str(round(elem/recall[i], 4)))
        print("Overall Correct: " + str(round(sum(diagonal)/matrixSum, 4)))
        
    def targetVector(self, key):
        return self.targetMatrix[key]

    def updateConfusionMatrix(self, key, outputs):
        maxIndex = 0
        maxValue = 0
        for i in range(len(outputs)):
            if maxValue < outputs[i]:
                maxIndex = i
                maxValue = outputs[i]
        self.confusionMatrix[key][maxIndex] = self.confusionMatrix[key][maxIndex] + 1

    def inputMagnitude(self):
        return self.inputMagX * self.inputMagY

    def outputMagnitude(self):
        return self.outputMag
