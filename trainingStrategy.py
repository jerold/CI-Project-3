#!/usr/bin/python

import random
import math
import patternSet
import Network

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
        self.genome = [[float(random.randrange(geneMin*10000, geneMax*10000))/10000 for _ in range(n)] for n in Member.genomeTemplate]
        if includeStrategyParameters:
            self.genome = self.genome + [[float(random.randrange(strategyMax*10000))/10000 for _ in range(n)] for n in Member.genomeTemplate]
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
        self.populationSize = 0
        self.alphas = [] # a list of the best members from each generation current best is [-1]

    @classmethod
    def getTrainingStrategyOfType(self, type=3):
        if type == TrainingStrategyType.EvolutionStrategy:
            return EvolutionStrategy()
        elif type == TrainingStrategyType.GeneticAlgorithm:
            return GeneticAlgorithm()
        elif type == TrainingStrategyType.DifferentialGA:
            return DifferentialGA()

    def updateMemberFitness(self, error):
        self.population[self.currentMember].adjustFitness(error)
        return 0

    def fitnessThresholdMet(self):
        if len(self.alphas) < 1:
            return False
        if self.alphas[-1].fitness < self.fitnessThreshold:
            return True
        return False

    def moreMembers(self):
        if self.currentMember < len(self.population):
            return True
        return False

    def continueToNextMember(self):
        self.currentMember = self.currentMember + 1

    def continueToNextGeneration(self):
        parents = self.select()
        child = self.crossover(parents)
        child = self.mutate(child)
        self.repopulate(parents + child)

        self.resetPopulationFitness()
        self.generation += self.generation
        self.currentMember = 0

    def initPopulation(self, pop, gRange, sParams, sMax):
        self.populationSize = pop
        self.population = []
        for p in range(pop):
            self.population.append(Member(gRange[0], gRange[-1], sParams, sMax))
        print("Member Genome Sample:")
        print("[" + ", ".join("[" + str(len(a)) + "]" for a in self.population[0].genome) + "]")
        # print("[" + ", ".join("[" + " ".join(str(b) for b in a) + "]" for a in self.population[0].genome) + "]")
        self.currentMember = 0

    def mutation(self):
        member = self.population[0]
        numberOfElements = len(member) * len(member[0])
        probability = 1 / float(numberOfElements)
        elem = random.random(0,numberOfElements) * probability
        choice = random.uniform(0, 1)
        diff = math.abs(elem - choice)
        if diff > probability:
            return False
        return True

    def epsilon(self):
        """epsilon"""
        return 0.15

    def resetPopulationFitness(self):
        """Between each generation fitness is reset"""
        for member in self.population:
            member.fitness = 0.0

    def getCurrentMemberWeightsForNeuron(self, neuronNumber):
        """Get method Neurons use to fetch their coorisponding weights from the current member's genome"""
        return self.population[self.currentMember].getGenesAtPosition(neuronNumber)

    def setCurrentMemberWeightsForNeuron(self, neuronNumber, weights):
        """Set method Neurons use to change their coorisponding weights within the current member's genome"""
        return self.population[self.currentMember].setGenesAtPosition(neuronNumber, weights)

    def select(self):
        """Returns a list of parents chosen for crossover"""
        raise("Instance of an Abstract Class... Bad Juju!")

    def crossover(self):
        """Returns a list of newly minted children"""
        raise("Instance of an Abstract Class... Bad Juju!")

    def mutate(self):
        """Go through all members of the population mutating at chance"""
        raise("Instance of an Abstract Class... Bad Juju!")

    def evaluateFitness(self):
        raise("Instance of an Abstract Class... Bad Juju!")

    def repopulate(self):
        """Given the current population and the population of children, combine to produce the next Generation"""
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
        """For the """
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
        return child

    def mutate(self, child):
        for gene in child:
            for elem in gene:
                if self.mutation():
                    if random.choice([True, False]):
                        elem += self.epsilon()
                    else:
                        elem -= self.epsilon()
        return child

    def evaluateFitness(self, child):
        fitness = 0
        for pattern in patternSet.patterns:
            Network.Layer.setInputs(Network.Net[0], pattern['p'])
            Network.Layer.feedforward(Network.Net[0])
            fitness += Network.Net.calculateConvError(Network.Net, pattern['t'])
        child.fitness = fitness

    def repopulate(self, contendors):
        bestFit = 0
        nextFit = 0
        for member in contendors:
            if member.fitness > bestFit:
                bestFitMember = member
            elif member.fitness > nextFit:
                nextFitMember = member
        self.population.append(bestFitMember)
        self.population.append(nextFitMember)



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


# if __name__=="__main__":
#     ts = TrainingStrategy.getTrainingStrategyOfType(TrainingStrategyType.GeneticAlgorithm)
#     Member.genomeTemplate = [3, 3, 3, 3, 4, 4]
#     memberCount = 10
#     ts.initPopulation(10, range(-5, 5), True, memberCount)
#     for i in range(memberCount):
#         print(ts.population[i].genome)
#     print(ts.getCurrentMemberWeightsForNeuron(1))
#     ts.setCurrentMemberWeightsForNeuron(1, [0.0, 1.1, 2.2])
#     print(ts.getCurrentMemberWeightsForNeuron(1))
#     ts.population[0].adjustFitness(4.50)
#     ts.population[0].adjustFitness(5.50)
#     print(ts.population[0].fitness)




