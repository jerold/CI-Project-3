#!/usr/bin/python
# Network.py


import time
import trainingStrategy as TS
from patternSet import PatternSet
import math


class NetLayerType:
    """Enum for Layer Type"""
    Input, Hidden, Output = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Input:"I",
            self.Hidden:"H",
            self.Output:"O"}[x]


class PatternType:
    """Enum for Pattern Type ( Also used as Net running Mode)"""
    Train, Test, Validate = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Train:"Train",
            self.Test:"Test",
            self.Validate:"Validate"}[x]


def sigmoidal(parameter):
    """Activation Funtion used by the Neurons during feed forward"""
    return math.tanh(parameter)


def outputError1(p, q):
    """Combined sum of the difference between two vectors"""
    errSum = 0.0
    for i in range(len(p)):
        errSum = errSum + math.fabs(p[i] - q[i])
    return errSum


def outputError(target, inVals):
    """Simple Correct or incorrect rating for the result"""
    maxIndex = 0
    maxValue = 0
    for i in range(len(inVals)):
        if maxValue < inVals[i]:
            maxIndex = i
            maxValue = inVals[i]
    if target[maxIndex] > 0:
        return 0
    return outputError1(target, inVals)


def vectorizeMatrix(p):
    """Turns a 2D matrix into a vector by appending the rows to one another"""
    if isinstance(p[0], list):
        v = []
        for i in p:
            v = v + i
        return v
    else:
        return p


def genomeTemplateFromArchitecture(inCount, hiddenArch, outCount):
    """The Genome contains all weights within the net, and is therefore constructed from the net's architecture"""
    print("Arch: [" + str(inCount) + "][" + "][".join(str(n) for n in hiddenArch) + "][" + str(outCount) + "]")
    tempGenomeTemp = [inCount for _ in range(hiddenArch[0])]
    #print("[" + str(inCount) + "]x" + str(hiddenArch[0]))
    for h in range(1, len(hiddenArchitecture)):
        tempGenomeTemp = tempGenomeTemp + [hiddenArchitecture[h-1] for _ in range(hiddenArchitecture[h])]
        #print("[" + str(hiddenArchitecture[h-1]) + "]x" + str(hiddenArchitecture[h]))
    tempGenomeTemp = tempGenomeTemp + [hiddenArchitecture[-1] for _ in range(outCount)]
    #print("[" + str(hiddenArchitecture[-1]) + "]x" + str(outCount))
    #print(tempGenomeTemp)
    return tempGenomeTemp


class Net:
    """The net is standard for MLP and all Genetic Algorithms.  It is composed of Layers
    containing neurons, which each has a collection of weights.  Feedforward is accomplished
    in a link list style from input layer to output layer.  The net can be configured to run
    in Training or Testing mods."""

    def __init__(self, patternSet, hiddenArch):
        Neuron.idIterator = 0
        Neuron.inputNeuronCount = 0

        self.layers = [Layer(NetLayerType.Input, None, patternSet.inputMagnitude())]
        for elem in hiddenArch:
            self.layers.append(Layer(NetLayerType.Hidden, self.layers[-1], elem))
        self.layers.append(Layer(NetLayerType.Output, self.layers[-1], patternSet.outputMagnitude()))
        self.patternSet = patternSet
        self.absError = 100
        
    # Run is where the magic happens. Training Testing or Validation mode is indicated and
    # the coorisponding pattern set is loaded and ran through the network
    # At the end Error is calculated
    def run(self, mode, startIndex, endIndex):
        """Run all patterns within the specified range, populate the confusion matrix if Testing
        train or pass on fitness information to the GA if in train mode."""
        patterns = self.patternSet.patterns
        Net.trainingStrategy.trainingMode = mode
        Net.trainingStrategy.patternCount = endIndex - startIndex
        print("Mode[" + PatternType.desc(mode) + ":" + str(endIndex - startIndex) + "]")
        startTime = time.time()

        if mode == PatternType.Train:
            while not Net.trainingStrategy.fitnessThresholdMet():
                # print("Generation[" + str(Net.trainingStrategy.generation) + "]")
                while Net.trainingStrategy.moreMembers():
                    self.layers[NetLayerType.Input].fetchNeuronWeightsForCurrentMember()
                    correctCategories = []
                    for i in range(startIndex, endIndex):
                        # print("G[" + str(Net.trainingStrategy.generation) + "] M[" + str(Net.trainingStrategy.currentMember) + "] P[" + str(i) + "]")
                        # Run each Pattern Through each member configuration, updating member weights with each pass
                        self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
                        self.layers[NetLayerType.Input].feedForward()
                        #print("Pattern Error: " + str(outputError(self.patternSet.targetVector(patterns[i]['t']), self.layers[-1].getOutputs())))
                        patternError = outputError(self.patternSet.targetVector(patterns[i]['t']), self.layers[-1].getOutputs())
                        if patternError == 0 and patterns[i]['t'] not in correctCategories:
                            correctCategories.append(patterns[i]['t'])
                        Net.trainingStrategy.updateMemberFitness(patternError/self.patternSet.counts[patterns[i]['t']])
                    if Net.trainingStrategy.runningChildren:
                        Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].categoryCoverage = correctCategories
                        Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].successFeedback(self.patternSet.combinedTargetVector(correctCategories))
                        fitness = Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].fitness
                        fitness = fitness/(endIndex - startIndex)
                        fitness = fitness/(len(correctCategories))
                        Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].fitness = fitness
                        # print("G[" + str(Net.trainingStrategy.generation) + "] C[" + str(Net.trainingStrategy.currentChildMember) + "] F[" + str(round(Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].fitness, 3)) + "] " + " ".join(str(c) for c in correctCategories))
                    else:
                        Net.trainingStrategy.population[Net.trainingStrategy.currentMember].categoryCoverage = correctCategories
                        Net.trainingStrategy.population[Net.trainingStrategy.currentMember].successFeedback(self.patternSet.combinedTargetVector(correctCategories))
                        fitness = Net.trainingStrategy.population[Net.trainingStrategy.currentMember].fitness
                        fitness = fitness/(endIndex - startIndex)
                        fitness = fitness/(len(correctCategories))
                        Net.trainingStrategy.population[Net.trainingStrategy.currentMember].fitness = fitness
                        # print("G[" + str(Net.trainingStrategy.generation) + "] M[" + str(Net.trainingStrategy.currentMember) + "] F[" + str(round(Net.trainingStrategy.population[Net.trainingStrategy.currentMember].fitness, 3)) + "] " + " ".join(str(c) for c in correctCategories))
                    Net.trainingStrategy.continueToNextMember()
                Net.trainingStrategy.continueToNextGeneration()
        else:
            self.layers[NetLayerType.Input].fetchNeuronWeightsForCurrentMember()
            for i in range(startIndex, endIndex):
                self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
                self.layers[NetLayerType.Input].feedForward()
                # print("Pattern: " + str(i))
                # print("Output: " + str(self.layers[-1].getOutputs()))
                # print("Target: " + str(self.patternSet.targetVector(patterns[i]['t'])))
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[-1].getOutputs())
        endTime = time.time()
        print("Run Time: [" + str(round(endTime-startTime, 2)) + " sec]")
                
    def calculateConvError(self, input):
        error = 0
        for i, neuron in enumerate(self.layers[-1].neurons):
            error += abs(input[i] - neuron.output)
        return error



class Layer:
    """A linked list, each layer contains neurons which contain weights."""

    def __init__(self, layerType, prevLayer, neuronCount):
        self.layerType = layerType
        self.prev = prevLayer
        if prevLayer != None:
            prevLayer.next = self
        self.next = None
        self.neurons = []
        for n in range(neuronCount):
            self.neurons.append(Neuron(self))

    def setInputs(self, inputVector):
        """Assign input values to the layer's neuron inputs"""
        if len(inputVector) != len(self.neurons):
            raise NameError('Input dimension of network does not match that of pattern!')
        for p in range(len(self.neurons)):
            self.neurons[p].input = float(inputVector[p])

    def getOutputs(self):
        """return a vector of this Layer's Neuron outputs"""
        out = []
        for neuron in self.neurons:
            out.append(neuron.output)
        return out

    def fetchNeuronWeightsForCurrentMember(self):
        """Typically called once for each member, this updates our neuron weights so we aren't
        continuously fetching them from the Training Strategy"""
        if self.prev:
            for neuron in self.neurons:
                neuron.weights = neuron.getMyWeights()['genes']
        if self.next:
            self.next.fetchNeuronWeightsForCurrentMember()

    def feedForward(self):
        """Each layer behaves a little differently here, but this for a given layer type
        we perform the appropreate feedforward steps (passing neuron input to output), and
        then tell the layer behind this one to feedforward"""
        if self.layerType == NetLayerType.Input:
            # Input Layer feeds all input to output with no work done
            for neuron in self.neurons:
                neuron.output = neuron.input
            self.next.feedForward()
        elif self.layerType == NetLayerType.Hidden:
            # net input is passed through a sigmoidal activation function to produce output
            prevOutputs = self.prev.getOutputs()
            for neuron in self.neurons:
                neuron.inputSum = 0.0
                for weight in neuron.weights:
                    for output in prevOutputs:
                        neuron.inputSum = neuron.inputSum + output * weight
                neuron.output = neuron.activate(neuron.inputSum)
            self.next.feedForward()
        elif self.layerType == NetLayerType.Output:
            # Linear Combination of Hidden layer outputs and associated weights
            prevOutputs = self.prev.getOutputs()
            for neuron in self.neurons:
                neuron.inputSum = 0.0
                for weight in neuron.weights:
                    for output in prevOutputs:
                        neuron.inputSum = neuron.inputSum + output * weight
                neuron.output = neuron.inputSum



# Neuron contains inputs and outputs and depending on the type will use
# weights or centers in calculating it's outputs.  Calculations are done
# in the layer as function of the neuron is tied to the layer it is contained in
class Neuron:
    """Contains inputs and outputs. Everything else in the neuron is provided to facilitate
    the passing and translation of input to output"""

    idIterator = 0
    inputNeuronCount = 0
    
    def __init__(self, layer):
        if layer.layerType == NetLayerType.Input:
            Neuron.inputNeuronCount = Neuron.inputNeuronCount + 1
        self.id = Neuron.idIterator
        Neuron.idIterator = Neuron.idIterator + 1

        # print("Neuron[" + str(self.id) + "]")
        self.layer = layer
        self.input = 0.00
        self.output = 0.00
        self.weights = []
        self.weightDeltas = []
        if layer.prev:
            self.weights = self.getMyWeights()['genes']
            for w, weight in enumerate(self.weights):
                self.weightDeltas.append(0.0)

    @classmethod
    def getWeights(self, neuronNumber):
        """Communicates with the Net's Training Strategy to fetch the current member weights for a given neuron"""
        if neuronNumber-Neuron.inputNeuronCount < 0:
            raise("I pitty the fool who tries to get weights for an input neuron.")
        # print("Weights for [" + str(neuronNumber) + ":" + str(neuronNumber-Neuron.inputNeuronCount) + "]")
        # print(Net.trainingStrategy.getCurrentMemberWeightsForNeuron(neuronNumber-Neuron.inputNeuronCount))
        return Net.trainingStrategy.getCurrentMemberWeightsForNeuron(neuronNumber-Neuron.inputNeuronCount)

    def getMyWeights(self):
        """Method returns a dictionary containing the 'genes' vector, and strategy parameters' vector (if applicable)"""
        return Neuron.getWeights(self.id)

    def activate(self, inputSum):
        """With your powers combined!"""
        return sigmoidal(inputSum)



#Main
if __name__=="__main__":
    # Batch:
    allDataTypes = ['data/ionosphere/ionosphere.json',
                    'data/block/pageblocks.json',
                    'data/heart/heart.json',
                    'data/glass/glass.json',
                    'data/car/car.json',
                    'data/seeds/seeds.json',
                    'data/wine/wine.json',
                    'data/yeast/yeast.json',
                    'data/zoo/zoo.json',
                    'data/iris/iris.json']

    # Single:
    # allDataTypes = ['data/ionosphere/ionosphere.json']
    # allDataTypes = ['data/block/pageblocks.json']
    # allDataTypes = ['data/heart/heart.json']
    # allDataTypes = ['data/glass/glass.json']
    # allDataTypes = ['data/car/car.json']
    # allDataTypes = ['data/seeds/seeds.json']
    # allDataTypes = ['data/wine/wine.json']
    # allDataTypes = ['data/yeast/yeast.json']
    # allDataTypes = ['data/zoo/zoo.json']
    # allDataTypes = ['data/iris/iris.json']

    # Batch:
    # strategies = [TS.TrainingStrategyType.EvolutionStrategy, TS.TrainingStrategyType.GeneticAlgorithm]

    # Single:
    # strategies = [TS.TrainingStrategyType.EvolutionStrategy]
    strategies = [TS.TrainingStrategyType.GeneticAlgorithm]
    # strategies = [TS.TrainingStrategyType.DifferentialGA]

    trainPercentage = 0.8
    maxGenerations = 4 #40
    populationSize = 3 #20
    runsPerDataSet = 2 #10

    #hiddenArchitecture = [14] # each hidden layer is a new index in this list, it's value = number of neurons in that layer
    for dataSet in allDataTypes:
        for strat in strategies:
            p = PatternSet(dataSet)
            for run in range(runsPerDataSet):
                print("\nData Set: (" + str(dataSet) + ") Run: " + str(run) + " Strategy: " + str(TS.TrainingStrategyType.desc(strat)))

                if run == 0:
                    p.initCombinedConfusionMatrix()
                hiddenArchitecture = [2*len(p.patterns[0]['p'])] # each hidden layer is a new index in this list, it's value = number of neurons in that layer
                TS.Member.genomeTemplate = genomeTemplateFromArchitecture(len(p.patterns[0]['p']), hiddenArchitecture, len(p.targets))
                Net.trainingStrategy = TS.TrainingStrategy.getTrainingStrategyOfType(strat)
                Net.trainingStrategy.maxGenerations = maxGenerations
                Net.trainingStrategy.initPopulation(populationSize, (-1.0, 1.0))
                n = Net(p, hiddenArchitecture)

                n.run(PatternType.Train, 0, int(p.count*trainPercentage))
                n.run(PatternType.Test, int(p.count*trainPercentage), p.count)
                # n.run(PatternType.Train, 0, p.count)
                # n.run(PatternType.Test, 0, p.count)
                p.printStats()
            p.saveConfusionMatrix("records/"+str(dataSet.split("/")[1])+"-"+str(TS.TrainingStrategyType.desc(strat))+".csv")
    print("Done")