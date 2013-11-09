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

def recordItems(recordString):
    with open('Strategy.log', 'a') as file:
        file.write(recordString + '\n')        

class Member():
    memberIdInc = 0
    genomeTemplate = [] # example [3, 3, 3, 3, 4, 4] Input layer has 3 nodes, Hidden has 4, output has 2

    def __init__(self, geneMin, geneMax, includeStrategyParameters, strategyMax):
        self.id = Member.memberIdInc
        Member.memberIdInc = Member.memberIdInc + 1
        self.genome = [[float(random.randrange(geneMin*10000, geneMax*10000))/10000 for _ in range(n)] for n in Member.genomeTemplate]
        self.sigmas = []
        if includeStrategyParameters:
            self.sigmas = [[float(random.randrange(strategyMax*10000))/10000-(strategyMax/2) for _ in range(n)] for n in Member.genomeTemplate]
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
        self.fitnessThreshold = .9
        self.population = []
        self.populationSize = 0

        self.trainingMode = Network.PatternType.Train
        self.lam = 1
        self.useSigmas = False
        self.sigmaMax = 1
        self.currentChildMember = 0
        self.childPopulation = []
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
        return self.population[self.currentMember].adjustFitness(error)

    def fitnessThresholdMet(self):
        if self.generation > 600:
            return True
        if len(self.alphas) < 1:
            return False
        if self.alphas[0].fitness <= self.fitnessThreshold:
            return True
        # self.resetPopulationFitness()
        return False

    def averageFitness(self):
        return sum(x.fitness for x in self.population)/len(self.population)

    def moreMembers(self):
        if self.currentMember < len(self.population):
            return True
        return False

    def continueToNextMember(self):
        self.currentMember = self.currentMember + 1

    def continueToNextGeneration(self):
        print("Average Fitness: " + str(int(self.averageFitness())))
        parents = self.select()
        for p in parents:
            print(str(p[0].fitness) + " " + str(p[1].fitness))
        self.childPopulation = self.crossover(parents)
        self.childPopulation = self.mutate(child)
        self.repopulate()

        self.generation = self.generation + 1
        self.currentMember = 0
        self.currentChildMember = 0
        self.childPopulation = []

    def initPopulation(self, pop, gRange):
        self.populationSize = pop
        if self.lam <= 1:
            self.lam = int(self.lam*self.populationSize)
        self.population = []
        for p in range(pop):
            self.population.append(Member(gRange[0], gRange[-1], self.useSigmas, self.sigmaMax))
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
        if self.trainingMode == Network.PatternType.Test:
            return self.alphas[0].getGenesAtPosition(neuronNumber)
        return self.population[self.currentMember].getGenesAtPosition(neuronNumber)

    def setCurrentMemberWeightsForNeuron(self, neuronNumber, weights):
        """Set method Neurons use to change their coorisponding weights within the current member's genome"""
        if self.trainingMode == Network.PatternType.Test:
            return self.alphas[0].setGenesAtPosition(neuronNumber)
        return self.population[self.currentMember].setGenesAtPosition(neuronNumber, weights)

    def select(self):
        """Returns a list of parents chosen for crossover"""
        raise("Instance of an Abstract Class... Bad Juju!")

    def crossover(self, parents):
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
        self.lam = 1.0
        self.strongerParentPreference = .5
        self.runningChildren = False
        self.fitnessThreshold = 10
        self.useSigmas = True
        self.sigmaMax = .001
        self.childSuccess = 0.0
        self.highestCurrentMemberId = 0

    def moreMembers(self):
        """In order to calulate fitness on the chilren we'll do selection, crossover, and mutation
        at the end of regular pop run, and then continue for the next set"""
        if not self.runningChildren and self.currentMember < len(self.population):
            return True
        if not self.runningChildren:
            # Once all current members have been evaluated, produce children mutate and calculate fitness for them
            parents = self.select()
            self.childPopulation = self.crossover(parents)
            self.mutate()
            self.runningChildren = True
            if len(self.childPopulation) > 0:
                return True
            return False
        if self.runningChildren and self.currentChildMember < len(self.childPopulation):
            return True
        return False
    
    def continueToNextMember(self):
        if not self.runningChildren:
            self.currentMember = self.currentMember + 1
        else:
            self.currentChildMember = self.currentChildMember + 1
        
    def continueToNextGeneration(self):
        self.repopulate()
        #print("G:" + str(self.generation) + " F[" + ", ".join(str(int(m.fitness)) for m in self.population) + "] Alph:" + str(int(self.alphas[0].fitness)) + " Avg: " + str(int(self.averageFitness())) + " P:" + str(round(self.childSuccess, 3)))
        print("G:" + str(self.generation) + " Alph:" + str(int(self.alphas[0].fitness)) + " Avg: " + str(int(self.averageFitness())) + " P:" + str(round(self.childSuccess, 3)))
        
        self.generation = self.generation + 1
        # self.currentMember = 0
        self.runningChildren = False
        self.currentChildMember = 0
        self.childPopulation = []

    def updateMemberFitness(self, error):
        if not self.runningChildren:
            return self.population[self.currentMember].adjustFitness(error)
        return self.childPopulation[self.currentChildMember].adjustFitness(error)

    def getCurrentMemberWeightsForNeuron(self, neuronNumber):
        """Get method Neurons use to fetch their coorisponding weights from the current member's genome"""
        if self.trainingMode == Network.PatternType.Test:
            return self.alphas[0].getGenesAtPosition(neuronNumber)
        if not self.runningChildren:
            return self.population[self.currentMember].getGenesAtPosition(neuronNumber)
        return self.childPopulation[self.currentChildMember].getGenesAtPosition(neuronNumber)

    def setCurrentMemberWeightsForNeuron(self, neuronNumber, weights):
        """Set method Neurons use to change their coorisponding weights within the current member's genome"""
        if self.trainingMode == Network.PatternType.Test:
            return self.alphas[0].setGenesAtPosition(neuronNumber)
        if not self.runningChildren:
            return self.population[self.currentMember].setGenesAtPosition(neuronNumber)
        return self.childPopulation[self.currentChildMember].setGenesAtPosition(neuronNumber)

    def select(self):
        # elites = self.population[:self.populationSize/2]
        moms = random.sample(self.population, self.lam)
        dads = random.sample(self.population, self.lam)
        #for p in parents:
        #    print(str(p[0].fitness) + " " + str(p[1].fitness))
        return list(zip(moms, dads))

    def crossover(self, parents):
        #Uniform Crossover, produces 1 child per pair of parents
        children = []
        #We use highestCurrentMemberId to check which of the members of the next generation are children of this generation
        self.highestCurrentMemberId = Member.memberIdInc
        #print(len(parents))
        for pair in parents:
            pair = list(pair)
            pair.sort(key=lambda x: x.fitness, reverse=False)
            child = Member(0, 1, self.useSigmas, self.sigmaMax)
            # by gene we also mean sigmas as the crossover for these is the same
            for g, gene in enumerate(child.genome):
                for w, singleWeight in enumerate(gene):
                    if random.random() <= self.strongerParentPreference:
                        child.genome[g][w] = pair[0].genome[g][w]
                    else:
                        child.genome[g][w] = pair[1].genome[g][w]
                    if random.random() <= self.strongerParentPreference:
                        child.sigmas[g][w] = pair[0].sigmas[g][w]
                    else:
                        child.sigmas[g][w] = pair[1].sigmas[g][w]
            children.append(child)
        return children

    def mutate(self):
        # Use 1/5 rule for sigma mutation
        sigmaMod = 1.0
        if self.childSuccess > 0.2:
            sigmaMod = 1.225
        else:
            sigmaMod = 0.816
        for child in self.childPopulation:
            for g, gene in enumerate(child.genome):
                for w, singleWeight in enumerate(gene):
                    child.sigmas[g][w] = child.sigmas[g][w] * sigmaMod
                    child.genome[g][w] = child.genome[g][w] + random.gauss(0, child.sigmas[g][w])
                    
    def evaluateFitness(self):
        return 0

    def repopulate(self):
        # (m+l)-ES
        #if self.generation > 0 and self.generation%20 == 0:
        #    print("Mass Extinction Event")
        #    self.initPopulation(self.populationSize, [-0.3, 0.3])
        #    self.childPopulation = self.crossover(zip(self.population, self.alphas[:self.lam]))
        #if self.population[0] in self.alphas:
        #    self.population = self.population[1:]
        oldAndNew = self.population + self.childPopulation
        oldAndNew.sort(key=lambda x: x.fitness, reverse=False)
        self.population = oldAndNew[:self.populationSize]
        if self.population[0] not in self.alphas and (len(self.alphas) == 0 or self.population[0].fitness < self.alphas[0].fitness):
            self.alphas.append(self.population[0])
            self.alphas.sort(key=lambda x: x.fitness, reverse=False)
        # print("New Pop: " + " ".join(str(int(m.fitness)) for m in self.population))
        # print("Alphas: " + " ".join(str(int(m.fitness)) for m in self.alphas))
        recordItems(", ".join(str(int(m.fitness)) for m in self.population) + ", " + str(self.childSuccess))
        self.childSuccess = 0.0
        for member in self.population:
            if member.id > self.highestCurrentMemberId:
                self.childSuccess = self.childSuccess + 1
        self.childSuccess = self.childSuccess/self.populationSize
                


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




