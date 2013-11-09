import random
import math
import time
from patternSet import PatternSet

eta = 1.00
learningRate = 0.1
momentum = 0.01

logWeights = False
logResults = False
logError = True

# Enum for Pattern Type ( Also used as Net running Mode)
class PatternType:
    Train, Test, Validate = range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Train:"Train",
            self.Test:"Test",
            self.Validate:"Validate"}[x]

# Enum for Layer Type
class NetLayerType:
    Input, Hidden, Output= range(3)

    @classmethod
    def desc(self, x):
        return {
            self.Input:"I",
            self.Hidden:"H",
            self.Output:"O"}[x]


# Weights are initialized to a random value between -0.3 and 0.3
def randomInitialWeight():
    return float(random.randrange(0, 6001))/10000 - .3


def sigmoidal(parameter):
    return math.tanh(parameter)


# combined sum of the difference between two vectors
def outputError(p, q):
    errSum = 0.0
    for i in range(len(p)):
        errSum = errSum + math.fabs(p[i] - q[i])
    return errSum


def derivative(parameter):
    return 1-(math.pow(math.tanh(parameter), 2))


def vectorizeMatrix(p):
    if isinstance(p[0], list):
        v = []
        for i in p:
            v = v + i
        return v
    else:
        return p

def euclidianDistance(p, q):
    sumOfSquares = 0.0
    for i in range(len(p)):
        sumOfSquares = sumOfSquares + ((p[i]-q[i])*(p[i]-q[i]))
    return math.sqrt(sumOfSquares)

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


class Network:
    def __init__(self, patternSet):
        self.iterations = 0
        inputLayer = Layer(NetLayerType.Input, None, patternSet.inputMagnitude())
        hiddenLayer = Layer(NetLayerType.Hidden, inputLayer, patternSet.inputMagnitude()*2)
        outputLayer = Layer(NetLayerType.Output, hiddenLayer, patternSet.outputMagnitude())
        self.layers = [inputLayer, hiddenLayer, outputLayer]
        self.output = []
        self.patternSet = patternSet
        self.absError = 100

    def run(self, mode, startIndex, endIndex):
        patterns = self.patternSet.patterns
        eta = 1.0
        errorSum = 0.0
        print("Mode[" + PatternType.desc(mode) + ":" + str(endIndex - startIndex) + "]")
        startTime = time.time()
        for i in range(startIndex, endIndex):
            #Initialize the input layer with input values from the pattern
            # Feed those values forward through the remaining layers, linked list style
            self.layers[NetLayerType.Input].setInputs(vectorizeMatrix(patterns[i]['p']))
            #self.layers[NetLayerType.Input].feedForward()
            if mode == PatternType.Train:
                #For training the final output weights are adjusted to correct for error from target
                #self.train(self.patternSet.targetVector(patterns[i]['p']), True)
                self.train(vectorizeMatrix(patterns[i]['p']), self.patternSet.targetVector(patterns[i]['t']), True)
            else:
                self.train(vectorizeMatrix(patterns[i]['p']), self.patternSet.targetVector(patterns[i]['t']), False)
                self.patternSet.updateConfusionMatrix(patterns[i]['t'], self.layers[NetLayerType.Output].getOutputs())
            # Each pattern produces an error which is added to the total error for the set
            # and used later in the Absolute Error Calculation
            outError = outputError(self.layers[NetLayerType.Output].getOutputs(), self.patternSet.targetVector(patterns[i]['t']))
            errorSum = errorSum + outError

    def train(self, dataSet, expectedOutput, train):
        output = []
        for _ in range(len(self.layers[-1].neurons)):
            output.append(0)
        #for data in dataSet[i]:
        self.iterations = 0
        for layer in self.layers:
            layer.feedForward()
        self.iterations += 1
        e = self.calculateConvError(expectedOutput)
        if train:
            while e > 0.01 and self.iterations < 100:
                self.backProp(expectedOutput)
                for layer in self.layers[1:]:
                    layer.feedForward()
                e = self.calculateConvError(expectedOutput)
                self.iterations += 1
                #print self.iterations
        for j, neuron in enumerate(self.layers[-1].neurons):
            output[j] = neuron.output
        if not train:
            print(output)

    def calculateConvError(self, input):
        error = 0
        for i, neuron in enumerate(self.layers[-1].neurons):
            error += abs(input[i] - neuron.output)
        return error

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


class Layer:
    def __init__(self, layerType, prevLayer, numNeurons):
        self.layerType = layerType
        self.prev = prevLayer
        if prevLayer:
            prevLayer.next = self
        self.next = None
        self.neurons = []
        for n in range(numNeurons):
            self.neurons.append(Neuron())
        if prevLayer:
            for n in self.neurons:
                for i in range(len(prevLayer.neurons)):
                    n.weight.append(randomInitialWeight())
                    n.weightChange.append(0)

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


class Neuron:
    def __init__(self):
        self.output = 0
        self.inputSum = 0
        self.error = 0
        self.weight = []
        self.weightChange = []

    def activate(self, inputSum):
        return sigmoidal(inputSum)



#Main
if __name__=="__main__":
    trainPercentage = 0.8
    #p = PatternSet('data/optdigits/optdigits-orig.json', trainPercentage, True)   # 32x32
    #p = PatternSet('data/letter/letter-recognition.json', trainPercentage, True)  # 1x16 # Try 1 center per attribute, and allow outputs to combine them
    #p = PatternSet('data/pendigits/pendigits.json', trainPercentage)        # 1x16 # same as above
    p = PatternSet('data/block/pageblocks.json', trainPercentage)            # 16x16 # Training set is very limited
    #p = PatternSet('data/adult/adult.json', trainPercentage)           # 16x16 # Training set is very limited
    #p = PatternSet('data/car/car.json', trainPercentage)        # 8x8
    #for e in range(1, 20):

    n = Network(p)
    n.run(PatternType.Train, 0, int(p.count*trainPercentage))
    n.run(PatternType.Test, int(p.count*trainPercentage), p.count)

    p.printConfusionMatrix()
    print("Done")
