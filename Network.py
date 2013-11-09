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

def outputError(p, q):
    """Combined sum of the difference between two vectors"""
    errSum = 0.0
    for i in range(len(p)):
        errSum = errSum + math.fabs(p[i] - q[i])
    return errSum

def outputError1(target, inVals):
    """Simple Correct or incorrect rating for the result"""
    maxIndex = 0
    maxValue = 0
    for i in range(len(inVals)):
        if maxValue < inVals[i]:
            maxIndex = i
            maxValue = inVals[i]
    # print(inVals)
    # print(target)
    # if target[maxIndex] > 0:
    #     print("Correct")
    # else:
    #     print("InCorrect")
    if target[maxIndex] > 0:
        return 0
    return 1

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
    def __init__(self, patternSet, hiddenArch):
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
        patterns = self.patternSet.patterns
        Net.trainingStrategy.trainingMode = mode
        Net.trainingStrategy.patternCount = endIndex - startIndex
        print("Mode[" + PatternType.desc(mode) + ":" + str(endIndex - startIndex) + "]")
        startTime = time.time()

        if mode == PatternType.Train:
            while not Net.trainingStrategy.fitnessThresholdMet():
                print("Generation[" + str(Net.trainingStrategy.generation) + "]")
                while Net.trainingStrategy.moreMembers():
                    self.layers[NetLayerType.Input].fetchNeuronWeightsForCurrentMember()
                    for i in range(startIndex, endIndex):
                        # print("G[" + str(Net.trainingStrategy.generation) + "] M[" + str(Net.trainingStrategy.currentMember) + "] P[" + str(i) + "]")
                        # Run each Pattern Through each member configuration, updating member weights with each pass
                        self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
                        self.layers[NetLayerType.Input].feedForward()
                        #print("Pattern Error: " + str(outputError(self.patternSet.targetVector(patterns[i]['t']), self.layers[-1].getOutputs())))
                        Net.trainingStrategy.updateMemberFitness(outputError(self.patternSet.targetVector(patterns[i]['t']), self.layers[-1].getOutputs()))
                    if Net.trainingStrategy.runningChildren:
                        Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].fitness = Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].fitness/(endIndex - startIndex)
                        print("G[" + str(Net.trainingStrategy.generation) + "] C[" + str(Net.trainingStrategy.currentChildMember) + "] F[" + str(round(Net.trainingStrategy.childPopulation[Net.trainingStrategy.currentChildMember].fitness, 3)) + "]")
                    else:
                        Net.trainingStrategy.population[Net.trainingStrategy.currentMember].fitness = Net.trainingStrategy.population[Net.trainingStrategy.currentMember].fitness/(endIndex - startIndex)
                        print("G[" + str(Net.trainingStrategy.generation) + "] M[" + str(Net.trainingStrategy.currentMember) + "] F[" + str(round(Net.trainingStrategy.population[Net.trainingStrategy.currentMember].fitness, 3)) + "]")                        
                    Net.trainingStrategy.continueToNextMember()
                Net.trainingStrategy.continueToNextGeneration()
        else:
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



#Layers are of types Input Hidden and Output.  
class Layer:
    def __init__(self, layerType, prevLayer, neuronCount):
        self.layerType = layerType
        self.prev = prevLayer
        if prevLayer != None:
            prevLayer.next = self
        self.next = None
        self.neurons = []
        for n in range(neuronCount):
            self.neurons.append(Neuron(self))

    # Assign input values to the layer's neuron inputs
    def setInputs(self, inputVector):
        if len(inputVector) != len(self.neurons):
            raise NameError('Input dimension of network does not match that of pattern!')
        for p in range(len(self.neurons)):
            self.neurons[p].input = float(inputVector[p])

    #return a vector of this Layer's Neuron outputs
    def getOutputs(self):
        out = []
        for neuron in self.neurons:
            out.append(neuron.output)
        return out

    def fetchNeuronWeightsForCurrentMember(self):
        if self.prev:
            for neuron in self.neurons:
                neuron.weights = neuron.getMyWeights()['genes']
        if self.next:
            self.next.fetchNeuronWeightsForCurrentMember()


    # Each Layer has a link to the next link in order.  Input values are translated from
    # input to output in keeping with the Layer's function
    def feedForward(self):
        if self.layerType == NetLayerType.Input:
            # Input Layer feeds all input to output with no work done
            for neuron in self.neurons:
                neuron.output = neuron.input
            self.next.feedForward()
        elif self.layerType == NetLayerType.Hidden:
            # RBF on the Euclidian Norm of input to center
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
        if neuronNumber-Neuron.inputNeuronCount < 0:
            raise("I pitty the fool who tries to get weights for an input neuron.")
        # print("Weights for [" + str(neuronNumber) + ":" + str(neuronNumber-Neuron.inputNeuronCount) + "]")
        # print(Net.trainingStrategy.getCurrentMemberWeightsForNeuron(neuronNumber-Neuron.inputNeuronCount))
        return Net.trainingStrategy.getCurrentMemberWeightsForNeuron(neuronNumber-Neuron.inputNeuronCount)

    def getMyWeights(self):
        """Method returns a dictionary containing the 'genes' vector, and strategy parameters' vector (if applicable)"""
        return Neuron.getWeights(self.id)

    def activate(self, inputSum):
        return sigmoidal(inputSum)


    
#Main
if __name__=="__main__":
    trainPercentage = 0.8
    attributeNeuronMultiplier = 2
    populationSize = 40
    
    #p = PatternSet('data/adult/adult.json', trainPercentage)        # Cases:32561 @ 1x16 # same as above
    p = PatternSet('data/car/car.json', trainPercentage)            # Cases:1382 @ 1x16 # same as above
    #p = PatternSet('data/pendigits/pendigits.json', trainPercentage)        # Cases:10992 @ 1x16 # same as above
    #p = PatternSet('data/block/pageblocks.json', trainPercentage)

    print("Weight Architecture:")
    # hiddenArchitecture = [len(p.patterns[0]['p'])*attributeNeuronMultiplier] # hidden layer is a new index in this list, value = number of neurons in that layer
    hiddenArchitecture = [12] # hidden layer is a new index in this list, value = number of neurons in that layer
    TS.Member.genomeTemplate = genomeTemplateFromArchitecture(len(p.patterns[0]['p']), hiddenArchitecture, len(p.targets))
    Net.trainingStrategy = TS.TrainingStrategy.getTrainingStrategyOfType(TS.TrainingStrategyType.GeneticAlgorithm)
    Net.trainingStrategy.initPopulation(populationSize, (-1.0, 1.0))
        
    n = Net(p, hiddenArchitecture)
    # n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    # n.run(PatternType.Test, int(p.count*trainPercentage), p.count)
    n.run(PatternType.Train, 0, 260)
    n.run(PatternType.Test, 0, 260)
    p.printConfusionMatrix()
    print("Done")
