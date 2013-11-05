#!/usr/bin/python

import random

class TrainingStrategyType:
    EvolutionStrategy, GeneticAlgorithm, DifferentialGA = range(3)

    @classmethod
    def desc(self, x):
        if x == 3:
            raise("Instance of an Abstract Class... Bad Juju!")
        return {self.EvolutionStrategy: "EvolutionStrategy",
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
        self.generation = 0
        self.currentMember = 0
        self.fitnessThreshold = 10
        self.population = []

    @classmethod
    def getTrainingStrategyOfType(self, type):
        if type == TrainingStrategyType.EvolutionStrategy:
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

    def fitnessThresholdMet(self):
        if self.evaluateFitness() > self.fitnessThreshold:
            return True
        return False

    def continueToNextGeneration(self):
        parents = self.select()
        self.crossover(parents)
        self.mutate()
        self.repopulate()

        self.resetPopulationFitness()
        self.generation = self.generation + 1
        self.currentMember = 0

    def initPopulation(self, pop, gRange, sParams, sMax):
        self.population = []
        for p in range(pop):
            self.population.append(Member(gRange[0], gRange[-1], sParams, sMax))
        self.currentMember = 0

    def mutation(self):
        numberOfElements = 0
        member = population[0]
        for gene in member:
            numberOfElements += 1
        probability = 1 \ float(numberOfElements)

    def resetPopulationFitness(self):
        for member in self.population:
            member.fitness = 0.0

    def getCurrentMemberWeightsForNeuron(self, neuronNumber):
        return self.population[self.currentMember].getGenesAtPosition(neuronNumber)

    def setCurrentMemberWeightsForNeuron(self, neuronNumber, weights):
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
        return random.sample(self.population, 2)

    def crossover(self, parents):
        child = []
        i = 0
        while i < len(parents[0]):
            parent1 = parents[i]
            parent2 = parents[i+1]
            i += 2
            for j, gene in enumerate(parent1):
                if j % 2 == 0:
                    child.append(parent1[j])
                else:
                    child.append(parent2[j])

    def mutate(self):
        if mutation:


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
    ts = TrainingStrategy.getTrainingStrategyOfType(TrainingStrategyType.GeneticAlgorithm)
    Member.genomeTemplate = [3, 3, 3, 3, 4, 4]
    memberCount = 10
    ts.initPopulation(10, range(-5, 5), True, memberCount)
    for i in range(memberCount):
        print(ts.population[i].genome)
    print(ts.getCurrentMemberWeightsForNeuron(1))
    ts.setCurrentMemberWeightsForNeuron(1, [0.0, 1.1, 2.2])
    print(ts.getCurrentMemberWeightsForNeuron(1))
    ts.population[0].adjustFitness(4.50)
    ts.population[0].adjustFitness(5.50)
    print(ts.population[0].fitness)




