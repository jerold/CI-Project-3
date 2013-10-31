#!/usr/bin/python

import random

class TrainingStrategyType:
    MLP, EvolutionStrategy, GeneticAlgorithm, DifferentialGA = range(4)

    @classmethod
    def desc(self, x):
        if x == 4:
            raise("Instance of an Abstract Class... Bad Juju!")
        return {self.MLP: "MLP",
                self.EvolutionStrategy: "EvolutionStrategy",
                self.GeneticAlgorithm: "GeneticAlgorithm",
                self.DifferentialGA: "DifferentialGA"}[x]



class Member():
    genomeTemplate = [] # example [3, 3, 3, 3, 4, 4] Input layer has 3 nodes, Hidden has 4, output has 2

    def __init__(self, geneMin, geneMax, includeStrategyParameters, strategyMax):
        self.genome = [[float(random.randrange(geneMin, geneMax)) for _ in range(n)] for n in Member.genomeTemplate]
        if includeStrategyParameters:
            self.genome = self.genome + [[float(random.randrange(strategyMax)) for _ in range(n)] for n in Member.genomeTemplate]
        self.fitness = 0.0

    def adjustFitness(self, value):
        self.fitness = self.fitness + value

    def getGenesAtPosition(self, neuronNumber):
        if len(self.genome) > len(Member.genomeTemplate):
            return {'genes':self.genome[neuronNumber], 'strategy parameters':self.genome[len(Member.genomeTemplate) + neuronNumber]}
        return {'genes':self.genome[neuronNumber]}

    # Used by the MLP during backprop
    def setGenesAtPosition(self, neuronNumber, values):
        if len(values) == len(self.genome[neuronNumber]):
            self.genome[neuronNumber] = values
        else:
            raise("Number of Genes does not match value set length")



class TrainingStrategy(object):
    def __init__(self):
        self.strategy = 4
        self.generation = 1
        self.fitnessThreshold = 10

    @classmethod
    def getTrainingStrategyOfType(self, type):
        if type == TrainingStrategyType.MLP:
            return MLP()
        elif type == TrainingStrategyType.EvolutionStrategy:
            return EvolutionStrategy()
        elif type == TrainingStrategyType.GeneticAlgorithm:
            return GeneticAlgorithm()
        elif type == TrainingStrategyType.DifferentialGA:
            return DifferentialGA()

    def atLastMember(self):
        if self.currentMember+1 ==  len(self.population):
            return True
        return False

    def updateFitness(self, error):
        self.population[self.currentMember].adjustFitness(error)
        return 0

    def continueToNextGeneration(self):
        if self.evaluateFitness() > self.fitnessThreshold:

            self.select()
            self.crossover()
            self.mutate()
            self.repopulate()

            self.resetPopulationFitness()
            self.generation = self.generation + 1
            self.currentMember = 0
            return True
        else:
            return False

    def initPopulation(self, pop, gRange, sParams, sMax):
        self.population = []
        for p in range(pop):
            self.population.append(Member(gRange[0], gRange[-1], sParams, sMax))
        self.currentMember = 0

    def resetPopulationFitness(self):
        for member in self.population:
            member.fitness = 0.0

    def getWeightsForNeuron(self, neuronNumber):
        return self.population[self.currentMember].getGenesAtPosition(neuronNumber)

    def setWeightsForNeuron(self, neuronNumber, weights):
        return self.population[self.currentMember].setGenesAtPosition(neuronNumber, weights)


    def select(self):
        raise("Instance of an Abstract Class... Bad Juju!")

    def crossover(self):
        raise("Instance of an Abstract Class... Bad Juju!")

    def mutate(self):
        raise("Instance of an Abstract Class... Bad Juju!")

    def evaluateFitness(self):
        raise("Instance of an Abstract Class... Bad Juju!")

    def repopulate(self):
        raise("Instance of an Abstract Class... Bad Juju!")



class MLP(TrainingStrategy):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.strategy = TrainingStrategyType.MLP

    def continueToNextGeneration(self):
        self.generation = self.generation + 1
        self.currentMember = 0

    def updateFitness(self, target):
        raise("Do not update fitness for MLP, have layers do backprop on their neurons")



class EvolutionStrategy(TrainingStrategy):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.strategy = TrainingStrategyType.EvolutionStrategy

    def select(self):
        return 0

    def crossover(self):
        return 0

    def mutate(self):
        return 0

    def evaluateFitness(self):
        return 0

    def repopulate(self):
        return 0



class GeneticAlgorithm(TrainingStrategy):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.strategy = TrainingStrategyType.GeneticAlgorithm

    def select(self):
        return 0

    def crossover(self):
        return 0

    def mutate(self):
        return 0

    def evaluateFitness(self):
        return 0

    def repopulate(self):
        return 0



class DifferentialGA(TrainingStrategy):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.strategy = TrainingStrategyType.DifferentialGA

    def select(self):
        return 0

    def crossover(self):
        return 0

    def mutate(self):
        return 0

    def evaluateFitness(self):
        return 0

    def repopulate(self):
        return 0


if __name__=="__main__":
    ts = TrainingStrategy.getTrainingStrategyOfType(TrainingStrategyType.MLP)
    Member.genomeTemplate = [3, 3, 3, 3, 4, 4]
    ts.initPopulation(10, range(-5, 5), True, 10)
    print(ts.population[0].genome)
    print(ts.getWeightsForNeuron(1))
    ts.setWeightsForNeuron(1, [0.0, 1.1, 2.2])
    print(ts.getWeightsForNeuron(1))
    ts.population[0].adjustFitness(4.50)
    ts.population[0].adjustFitness(5.50)
    print(ts.population[0].fitness)




