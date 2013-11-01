#!/usr/bin/python
# Network.py
import math
from patternSet import PatternSet

eta = 1.00
learningRate = 0.1
momentum = 0.01


def backProp(self, output):
    self.calculateErrors(output)
    #expectedOutput(output)
    self.updateWeights()


def calculateErrors(self, expoutput):
        for i, neuron in enumerate(self.layers[-1].neurons):
            output = neuron.output
            target = expoutput[i]
            neuron.error = (target - output)
        #hidden layer move backwards
        layer = self.layers[-1]
        while layer.prev:
            layer = layer.prev
            nextLayer = layer.next
            for j, neuron in enumerate(layer.neurons):
                sum = 0
                for n in nextLayer.neurons:
                    sum += n.weight[j] * n.error
                neuron.error = derivative(neuron.inputSum) * sum


def updateWeights(self):
        prevlayer = self.layers[-2]
        first = True
        while prevlayer.prev:
            if first:
                layer = self.layers[-1]
                prevlayer = prevlayer
                first = False
            else:
                layer = prevlayer
                prevlayer = layer.prev
            for neuron in layer.neurons:
                for k, weight in enumerate(neuron.weight):
                    neuron.weightChange[k] = learningRate * neuron.error * prevlayer.neurons[k].output + momentum * neuron.weightChange[k]
                    neuron.weight[k] += neuron.weightChange[k]

# feed forward from MLP
def feedForward(self):
    if self.layerType == NetLayerType.Input:
        # Input Layer feeds all input to output with no work done
        for neuron in self.neurons:
            neuron.output = neuron.input
    elif self.layerType == NetLayerType.Hidden:
        prevOutputs = self.prev.getOutputs()
        for neuron in self.neurons:
            neuron.inputSum = 0
            for weight in neuron.weight:
                for output in prevOutputs:
                    neuron.inputSum += output * weight
            neuron.output = learningRate * neuron.activate(neuron.inputSum)
    elif self.layerType == NetLayerType.Output:
        prevOutputs = self.prev.getOutputs()
        for neuron in self.neurons:
            neuron.inputSum = 0
            for weight in neuron.weight:
                for output in prevOutputs:
                    neuron.inputSum += output * weight
            neuron.output = neuron.inputSum


def calculateConvError(self, input):
    error = 0
    for i, neuron in enumerate(self.layers[-1].neurons):
        error += abs(input[i] - neuron.output)
    return error


def derivative(parameter):
    return 1-(math.pow(math.tanh(parameter), 2))