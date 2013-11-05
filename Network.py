#!/usr/bin/python
# Network.py

import time
import trainingStrategy as TS
# from trainingStrategy import Member
# from trainingStrategy import TrainingStrategy
# from trainingStrategy import TrainingStrategyType
from patternSet import PatternSet
import math

# Enum for Layer Type
class NetLayerType:
    Input, Hidden, Output = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Input:"I",
            self.Hidden:"H",
            self.Output:"O"}[x]

# Enum for Pattern Type ( Also used as Net running Mode)
class PatternType:
    Train, Test, Validate = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Train:"Train",
            self.Test:"Test",
            self.Validate:"Validate"}[x]




# combined sum of the difference between two vectors
def outputError(p, q):
    errSum = 0.0
    for i in range(len(p)):
        errSum = errSum + math.fabs(p[i] - q[i])
    return errSum

def vectorizeMatrix(p):
    if isinstance(p[0], list):
        v = []
        for i in p:
            v = v + i
        return v
    else:
        return p



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
        print("Mode[" + PatternType.desc(mode) + ":" + str(endIndex - startIndex) + "]")
        startTime = time.time()

        if mode == PatternType.Train:
            while not Net.trainingStrategy.fitnessThresholdMet():
                while Net.trainingStrategy.moreMembers():
                    for i in range(startIndex, endIndex):
                        # Run each Pattern Through each member configuration, updating member weights with each pass
                        print("G[" + str(Net.trainingStrategy.generation) + " M[" + str(Net.trainingStrategy.currentMember) + "] P:[" + str(i) + "]")

                        self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
                        self.layers[NetLayerType.Input].feedForward()
                        Net.trainingStrategy.updateMemberFitness(outputError(patterns[i]['t'], self.layers[-1].getOutputs()))
                    Net.trainingStrategy.continueToNextMember()
                Net.trainingStrategy.continueToNextGeneration()
        else:
            for i in range(startIndex, endIndex):
                self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
                self.layers[NetLayerType.Input].feedForward()
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[-1].getOutputs())

        endTime = time.time()
        print("Run Time: [" + str(endTime-startTime) + "sec]")
                
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
            self.neurons[p].input = inputVector[p]

    #return a vector of this Layer's Neuron outputs
    def getOutputs(self):
        out = []
        for neuron in self.neurons:
            out.append(neuron.output)
        return out

    # Each Layer has a link to the next link in order.  Input values are translated from
    # input to output in keeping with the Layer's function
    def feedForward(self):
        if self.layerType == NetLayerType.Input:
            # Input Layer feeds all input to output with no work done
            # for n, neuron in enumerate(self.neurons):
                # Do work
                # print("FF Neuron in Input Layer")
            self.next.feedForward()
        elif self.layerType == NetLayerType.Hidden:
            # RBF on the Euclidian Norm of input to center
            # for n, neuron in enumerate(self.neurons):
                # Do work
                # print("FF Neuron in Hidden Layer")
            self.next.feedForward()
        # elif self.layerType == NetLayerType.Output:
            # Linear Combination of Hidden layer outputs and associated weights
            # for n, neuron in enumerate(self.neurons):
                # Do work
                # print("FF Neuron in Output Layer")


    # Output Format
    def __str__(self):
        out = "  " + NetLayerType.desc(self.layerType) + "["
        for neuron in self.neurons:
            out = out + str(neuron)
        out = out + "]\n"
        return out



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
        self.weightDeltas = []
        if layer.prev:
            weights = self.getMyWeights()
            for w, weight in enumerate(weights):
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


    
#Main
if __name__=="__main__":
    trainPercentage = 0.8
    attributeNeuronMultiplier = 2
    populationSize = 10
    
    p = PatternSet('data/adult/adult.json', trainPercentage)        # 10992 @ 1x16 # same as above


    print("Weight Architecture:")
    hiddenArchitecture = [len(p.patterns[0]['p'])*attributeNeuronMultiplier] # hidden layer is a new index in this list, value = number of neurons in that layer
    genomeTemplate = [len(p.patterns[0]['p']) for _ in range(len(p.patterns[0]['p'])*attributeNeuronMultiplier)]
    print("[" + str(len(p.patterns[0]['p'])) + "]x" + str(len(p.patterns[0]['p'])*attributeNeuronMultiplier))
    for h in range(1, len(hiddenArchitecture)):
        genomeTemplate = genomeTemplate + list([hiddenArchitecture[h-1] for _ in range(hiddenArchitecture[h])])
        print("[" + str(hiddenArchitecture[h-1]) + "]x" + str(hiddenArchitecture[h]))
    genomeTemplate = genomeTemplate + list([hiddenArchitecture[-1] for _ in range(len(p.targets))])
    print("[" + str(hiddenArchitecture[-1]) + "]x" + str(len(p.targets)))


    TS.Member.genomeTemplate = genomeTemplate
    Net.trainingStrategy = TS.TrainingStrategy.getTrainingStrategyOfType(TS.TrainingStrategyType.GeneticAlgorithm)
    Net.trainingStrategy.initPopulation(populationSize, (-0.3, 0.3), False, 0)
        
    n = Net(p, hiddenArchitecture)
    n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    n.run(PatternType.Test, int(p.count*trainPercentage), p.count)
    p.printConfusionMatrix()
    print("Done")
