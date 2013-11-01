#!/usr/bin/python
# Network.py

import time
from trainingStrategy import TrainingStrategy
from trainingStrategy import TrainingStrategyType
from patternSet import PatternSet

# Enum for Layer Type
class NetLayerType:
    Input, Hidden, Output = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Input:"I",
            self.Hidden:"H",
            self.Output:"O"}[x]


    
class Net:
    def __init__(self, patternSet, hiddenArch, trainingStrategyType):
        trainingStrategy = TrainingStrategy.getTrainingStrategyOfType(trainingStrategyType)
        self.layers.append(Layer(NetLayerType.Input, None, patternSet.inputMagnitude()))
        for elem in hiddenArch:
            self.layer.append(Layer(NetLayerType.Hidden, self.layers[-1], elem))
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
        nextGenRequired = True
        while nextGenRequired:
            for i in range(startIndex, endIndex):
                self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
                self.layers[NetLayerType.Input].feedForward()
                if mode == PatternType.Train:
                    if trainingStrategy.atLastMember():
                        if trainingStrategy.strategy == TrainingStrategyType.MLP:
                            # do backprop
                            self.layers[-1].backPropagation(self.patternSet.targetVector(patterns[i]['t']))
                        trainingStrategy.continueToNextGeneration()
                        nextGenRequired = not trainingStrategy.fitnessThresholdMet()
                else:
                    self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[-1].getOutputs())
                    if trainingStrategy.atLastMember():
                        nextGenRequired = False
        endTime = time.time()
        print("Run Time: [" + str(endTime-startTime) + "sec]")
                


#Layers are of types Input Hidden and Output.  
class Layer:
    def __init__(self, layerType, prevLayer, neuronCount, trainingStrategy):
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
            for n, neuron in enumerate(self.neurons):
                # Do work
                print("FF Neuron in Input Layer")
            self.next.feedForward()
        elif self.layerType == NetLayerType.Hidden:
            # RBF on the Euclidian Norm of input to center
            for n, neuron in enumerate(self.neurons):
                # Do work
                print("FF Neuron in Hidden Layer")
            self.next.feedForward()
        elif self.layerType == NetLayerType.Output:
            # Linear Combination of Hidden layer outputs and associated weights
            for n, neuron in enumerate(self.neurons):
                # Do work
                print("FF Neuron in Output Layer")

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
    
    def __init__(self, layer):
        self.id = Neuron.idIterator
        self.layer = layer
        self.input = 0.00
        self.output = 0.00
        self.weightDeltas = []
        if layer.prev:
            weights = Net.trainingStrategy.getCurrentMemberWeightsForNeuron(self.id)
            for w, weight in enumerate(weights):
                self.weightDeltas.append(0.0)
        Neuron.idIterator += 1

    @classmethod
    def getWeights(neuronNumber):
        return Net.trainingStrategy.getCurrentMemberWeightsForNeuron(neuronNumber)

    def getWeights(self):
        return Neuron.getWeights(self.id)

    def setWeights(neuronNumber):
        return Net.trainingStrategy.setCurrentMemberWeightsForNeuron(neuronNumber)

    def setWeights(self):
        return Neuron.setWeights(self.id)


    
#Main
if __name__=="__main__":
    trainPercentage = 0.8
    p = PatternSet('data/pendigits/pendigits.json', trainPercentage)        # 10992 @ 1x16 # same as above
    n = Net(p)
    n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    n.run(PatternType.Test, int(p.count*trainPercentage), p.count)
    p.printConfusionMatrix()
    print("Done")
