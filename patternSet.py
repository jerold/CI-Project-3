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
            try:
                counts[pattern['t']] = 1
            except TypeError:
                counts[str(pattern['t'])] = 1
        else:
            try:
                counts[pattern['t']] = counts[pattern['t']] + 1
            except TypeError:
                counts[str(pattern['t'])] = counts[str(pattern['t'])] + 1
    targets.sort()
    try:
        print("Targets: [" + ", ".join(str(t) + "x" + str(counts[t]) for t in targets) + "]")
    except TypeError:
        pass
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
    confusionMatrix = {}
    correctness = []

    # Reads patterns in from a file, and puts them in their coorisponding set
    def __init__(self, fileName):
        with open(fileName) as jsonData:
            data = json.load(jsonData)
        self.name = fileName
            
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
        # keys = self.counts.keys()
        # minPatternCount = 9999
        # maxPatternCount = 0
        # patternTargetSets = {}
        # self.newPatterns = []
        # for key in keys:
        #     if minPatternCount > self.counts[key]:
        #         minPatternCount = self.counts[key]
        #     if maxPatternCount < self.counts[key]:
        #         maxPatternCount = self.counts[key]
        # for key in keys:
        #     patternTargetSets[key] = []
        #     for pat in self.patterns:
        #         if pat['t'] == key:
        #             patternTargetSets[key].append(pat)
        #     setMultiplier = 0
        #     while setMultiplier*len(patternTargetSets[key]) < maxPatternCount:
        #         if (setMultiplier+1)*len(patternTargetSets[key]) > maxPatternCount:
        #             addAmount = (setMultiplier+1)*len(patternTargetSets[key]) - maxPatternCount
        #             self.newPatterns = self.newPatterns + patternTargetSets[key][:addAmount]
        #         else:
        #             self.newPatterns = self.newPatterns + patternTargetSets[key]
        #         setMultiplier = setMultiplier + 1
        # self.patterns = self.newPatterns
        # self.count = len(self.patterns)
        #
        # targetsWithCounts = findUniqueTargets(self.patterns)
        # self.targets = targetsWithCounts['targets']
        # self.counts = targetsWithCounts['counts']

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
            try:
                self.confusionMatrix[key] = [0.0] * len(keys)
                self.targetMatrix[key] = [0] * len(keys)
                self.targetMatrix[key][index] = 1
                index = index + 1
            except TypeError:
                self.confusionMatrix[str(key)] = [0.0] * len(keys)
                self.targetMatrix[str(key)] = [0] * len(keys)
                self.targetMatrix[str(key)][index] = 1
                index = index + 1

        #init Class Level Confusion Matrix for Multi Run Stats
        if len(PatternSet.confusionMatrix) == 0:
            for key in keys:
                try:
                    PatternSet.confusionMatrix[key] = [0.0] * len(keys)
                except TypeError:
                    PatternSet.confusionMatrix[str(key)] = [0.0] * len(keys)
            PatternSet.correctness = []

        self.outputMag = len(keys)

    def initCombinedConfusionMatrix(self):
        PatternSet.confusionMatrix = {}
        PatternSet.correctness = []
        keys = self.targets
        keys.sort()
        for key in keys:
            try:
                PatternSet.confusionMatrix[key] = [0.0] * len(keys)
            except TypeError:
                PatternSet.confusionMatrix[str(key)] = [0.0] * len(keys)
        PatternSet.correctness = []

    def printCombinedCorrectness(self):
        if len(PatternSet.correctness) > 0:
            average = sum(PatternSet.correctness)/len(PatternSet.correctness)
            squaredDifferences = 0.0
            for val in PatternSet.correctness:
                squaredDifferences = squaredDifferences + (val - average)*(val - average)
            meanSquaredDifference = squaredDifferences/len(PatternSet.correctness)
            print("Standard Deviation of Correctness: " + str(round(math.sqrt(meanSquaredDifference), 4)))

    def printStats(self):
        print("\nConfusion Matrix")
        self.printConfusionMatrix(self.confusionMatrix)
        self.calcPrecisionAndRecall(self.confusionMatrix)
        if len(PatternSet.correctness) > 1:
            print("\nMulti-Run Combined Confusion Matrix")
            self.printConfusionMatrix(PatternSet.confusionMatrix)
            self.calcPrecisionAndRecall(PatternSet.confusionMatrix)
            self.printCombinedCorrectness()

    def printConfusionMatrix(self, confMatrix):
        keys = list(confMatrix.keys())
        keys.sort()
        for key in keys:
            printPatterns(confMatrix[key])

    def saveConfusionMatrix(self, p):
        keys = list(p.confusionMatrix.keys())
        keys.sort()
        with open('confusionMatrices.csv', 'a') as file:
            file.write(self.name + '\n')
            for key in keys:
                for i, elem in enumerate(self.confusionMatrix[key]):
                    file.write(str(elem))
                    if i < len(keys)-1:
                        file.write(',')
                file.write('\n')

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

    def calcPrecisionAndRecall(self, confMatrix):
        keys = list(confMatrix.keys())
        matrixSum = 0.0
        keys.sort()
        i = 0
        precision = []
        recall = []
        diagonal = []
        for key in keys:
            row = confMatrix[key]
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
        print("\nKey, Precision, Recall")
        for i, elem in enumerate(diagonal):
            if abs(precision[i]) > 0.0 and abs(recall[i]) > 0.0:
                print(str(keys[i]) + ", " + str(round(elem / precision[i], 4)) + ", " + str(round(elem/recall[i], 4)))
        PatternSet.correctness.append(sum(diagonal)/matrixSum)
        print("Overall Correct: " + str(round(sum(diagonal)/matrixSum, 4)))

    def targetVector(self, key):
        try:
            return self.targetMatrix[key]
        except TypeError:
            return self.targetMatrix[str(key)]

    def combinedTargetVector(self, keys):
        if len(keys) == 1:
            return self.targetMatrix[keys[0]]
        ctv = [t for t in self.targetMatrix[keys[0]]]
        for key in keys[1:]:
            tv = self.targetMatrix[key]
            for c, cat in enumerate(ctv):
                ctv[c] = tv[c] if tv[c] == 1 else ctv[c]
        return ctv

    def updateConfusionMatrix(self, key, outputs):
        maxIndex = 0
        maxValue = 0
        for i in range(len(outputs)):
            if maxValue < outputs[i]:
                maxIndex = i
                maxValue = outputs[i]
        self.confusionMatrix[key][maxIndex] = self.confusionMatrix[key][maxIndex] + 1
        PatternSet.confusionMatrix[key][maxIndex] = PatternSet.confusionMatrix[key][maxIndex] + 1

    def inputMagnitude(self):
        return self.inputMagX * self.inputMagY

    def outputMagnitude(self):
        return self.outputMag
