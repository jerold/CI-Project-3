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

def avgSigma(population):
    sigSum = 0.0
    sigCount = 0
    for member in population:
        ms = Network.vectorizeMatrix(member.sigmas)
        for singleSigma in ms:
            sigSum = sigSum + singleSigma
            sigCount = sigCount + 1
    return sigSum/sigCount

def memberVarience(population, alpha):
    diffSum = 0.0
    for member in population:
        mg = Network.vectorizeMatrix(member.genome)
        ag = Network.vectorizeMatrix(alpha.genome)
        diffSum = diffSum + Network.outputError(mg, ag)
    return diffSum/len(population)

def averageFitness(population):
    return sum(x.fitness for x in population)/len(population)



class Member():
    memberIdInc = 0
    genomeTemplate = [] # example [3, 3, 3, 3, 4, 4] Input layer has 3 nodes, Hidden has 4, output has 2

    def __init__(self, geneMin, geneMax, includeStrategyParameters, strategyMax):
        self.id = Member.memberIdInc
        Member.memberIdInc = Member.memberIdInc + 1
        self.genome = [[float(random.randrange(geneMin*10000, geneMax*10000))/10000 for _ in range(n)] for n in Member.genomeTemplate]
        self.highPerformers = [0 for _ in Member.genomeTemplate]
        self.sigmas = []
        if includeStrategyParameters:
            self.sigmas = [[float(random.randrange(strategyMax*10000))/10000-(strategyMax/2) for _ in range(n)] for n in Member.genomeTemplate]
        self.fitness = 0.0
        self.categoryCoverage = []

    def adjustFitness(self, value):
        self.fitness = self.fitness + value

    def successFeedback(self, feedbackVector):
        """The Feedback Vector represents the categories from which this member was able to choose correctly, therefore we will give preference to the cooresponding genes during combination"""
        for i, fbv in enumerate(feedbackVector):
            self.highPerformers[-1*len(feedbackVector) + i] = fbv
        # print("FV:" + str(feedbackVector))
        # print("HP:" + str(self.highPerformers))

    def getGenesAtPosition(self, neuronNumber):
        if len(self.genome) > len(Member.genomeTemplate):
            return {'genes':self.genome[neuronNumber], 'strategy parameters':self.genome[-1]}
        return {'genes':self.genome[neuronNumber]}

    # Used by the MLP during backprop
    def setGenesAtPosition(self, neuronNumber, values):
        if len(values) == len(self.genome[neuronNumber]):
            self.genome[neuronNumber] = values
        else:
            raise("Number of Genes does not match value set length")



class TrainingStrategy(object):
    def __init__(self):
        Member.memberIdInc = 0
        self.strategy = 4
        self.trainingMode = Network.PatternType.Train

        self.generation = 0

        self.runningChildren = False
        self.currentMember = 0
        self.population = []
        self.runningChildren = False
        self.currentChildMember = 0
        self.childPopulation = []
        self.alphas = [] # a list of the best members from each generation current best is [-1]
        self.highestCurrentMemberId = 0
        self.childSuccess = 0.0

        self.fitnessThreshold = .00005
        self.maxPopulationSize = 0
        self.lam = 1.0
        self.useSigmas = False
        self.sigmaMax = 1
        self.maxGenerations = 10

    @classmethod
    def getTrainingStrategyOfType(self, type=3):
        if type == TrainingStrategyType.EvolutionStrategy:
            return EvolutionStrategy()
        elif type == TrainingStrategyType.GeneticAlgorithm:
            return GeneticAlgorithm()
        elif type == TrainingStrategyType.DifferentialGA:
            return DifferentialGA()

    def initPopulation(self, pop, gRange):
        self.maxPopulationSize = pop
        self.population = []
        for p in range(pop):
            self.population.append(Member(gRange[0], gRange[-1], self.useSigmas, self.sigmaMax))
        if self.lam <= 1:
            self.lam = int(self.lam*self.maxPopulationSize)            
        print("Member Genome Sample:")
        print("[" + ", ".join("[" + str(len(a)) + "]" for a in self.population[0].genome) + "]")
        # print("[" + ", ".join("[" + " ".join(str(b) for b in a) + "]" for a in self.population[0].genome) + "]")
        self.currentMember = 0

    def avgSigma(self):
        return avgSigma(self.population)

    def memberVarience(self):
        return memberVarience(self.population, self.alphas[0])

    def averageFitness(self):
        return averageFitness(self.population)

    def epsilon(self):
        """epsilon"""
        return 0.15

    #---- Net Interface Methods ---------------------------------------------
    
    def fitnessThresholdMet(self):
        if self.generation > self.maxGenerations:
            return True
        if len(self.alphas) < 1:
            return False
        if self.alphas[0].fitness <= self.fitnessThreshold:
            return True
        return False

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
        if self.trainingMode == Network.PatternType.Test:
            self.currentAlphaMember = self.currentAlphaMember + 1
        else:
            if not self.runningChildren:
                self.currentMember = self.currentMember + 1
            else:
                self.currentChildMember = self.currentChildMember + 1
            
    def continueToNextGeneration(self):
        self.repopulate()
        #print("G:" + str(self.generation) + " F[" + ", ".join(str(int(m.fitness)) for m in self.population) + "] Alph:" + str(int(self.alphas[0].fitness)) + " Avg: " + str(int(self.averageFitness())) + " P:" + str(round(self.childSuccess, 3)))
        
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

    # def moreMembers(self):
    #     if self.currentMember < len(self.population):
    #         return True
    #     return False

    # def continueToNextMember(self):
    #     self.currentMember = self.currentMember + 1

    # def continueToNextGeneration(self):
    #     print("Average Fitness: " + str(int(self.averageFitness())))
    #     parents = self.select()
    #     for p in parents:
    #         print(str(p[0].fitness) + " " + str(p[1].fitness))
    #     self.childPopulation = self.crossover(parents)
    #     self.childPopulation = self.mutate(self.childPopulation)
    #     self.repopulate()

    #     self.generation = self.generation + 1
    #     self.currentMember = 0
    #     self.currentChildMember = 0
    #     self.childPopulation = []

    # def getCurrentMemberWeightsForNeuron(self, neuronNumber):
    #     """Get method Neurons use to fetch their coorisponding weights from the current member's genome"""
    #     if self.trainingMode == Network.PatternType.Test:
    #         return self.alphas[0].getGenesAtPosition(neuronNumber)
    #     return self.population[self.currentMember].getGenesAtPosition(neuronNumber)

    # def setCurrentMemberWeightsForNeuron(self, neuronNumber, weights):
    #     """Set method Neurons use to change their coorisponding weights within the current member's genome"""
    #     if self.trainingMode == Network.PatternType.Test:
    #         return self.alphas[0].setGenesAtPosition(neuronNumber)
    #     return self.population[self.currentMember].setGenesAtPosition(neuronNumber, weights)

    #---- Genetic Algorithm Methods -----------------------------------------

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
        self.useSigmas = True
        self.sigmaMax = .001

        self.strongerParentPreference = .5
        self.alphaCategoryMap = {}

    def select(self):
        moms = random.sample(self.population, self.lam)
        dads = random.sample(self.population, self.lam)
        for i in range(self.lam):
            yield [moms[i], dads[i]]

    def crossover(self, parents):
        #Uniform Crossover, produces 1 child per pair of parents
        #We use highestCurrentMemberId to check which of the members of the next generation are children of this generation
        self.highestCurrentMemberId = Member.memberIdInc
        #print(len(parents))
        for pair in parents:
            pair = list(pair)
            pair.sort(key=lambda x: x.fitness, reverse=False)
            child = Member(0, 1, self.useSigmas, self.sigmaMax)
            # by gene we also mean sigmas as the crossover for these is the same
            for g, gene in enumerate(child.genome):
                geneSuccessMod = self.strongerParentPreference*pair[0].highPerformers[g] - self.strongerParentPreference*pair[1].highPerformers[g]
                # print("Mod:" + str(geneSuccessMod) + "[" + str(pair[0].highPerformers[g]) + "][" + str(pair[1].highPerformers[g]) + "]")
                for w, singleWeight in enumerate(gene):
                    if random.random() <= self.strongerParentPreference + geneSuccessMod:
                        child.genome[g][w] = pair[0].genome[g][w]
                    else:
                        child.genome[g][w] = pair[1].genome[g][w]
                    if random.random() <= self.strongerParentPreference + geneSuccessMod:
                        child.sigmas[g][w] = pair[0].sigmas[g][w]
                    else:
                        child.sigmas[g][w] = pair[1].sigmas[g][w]
            self.childPopulation.append(child)
        return self.childPopulation

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
        oldAndNew = self.population + self.childPopulation
        oldAndNew.sort(key=lambda x: x.fitness, reverse=False)
        self.population = oldAndNew[:self.maxPopulationSize]
        if self.population[0] not in self.alphas and (len(self.alphas) == 0 or self.population[0].fitness < self.alphas[0].fitness):
            self.alphas.append(self.population[0])
            self.alphas.sort(key=lambda x: x.fitness, reverse=False)
        recordItems(", ".join(str(int(m.fitness)) for m in self.population) + ", " + str(self.childSuccess))
        self.childSuccess = 0.0
        for member in self.population:
            if member.id > self.highestCurrentMemberId:
                self.childSuccess = self.childSuccess + 1
        self.childSuccess = self.childSuccess/self.maxPopulationSize
        # print("G:" + str(self.generation) + " CC:[" + ", ".join(str(m.categoryCoverage) + ":" + str(round(m.fitness, 4)) for m in self.population) + "] AvgSig:" + str(round(self.avgSigma(), 3)) + " MemVar:" + str(round(self.memberVarience(), 4)) + " Alph:" + str(round(self.alphas[0].fitness, 4)) + " Avg: " + str(round(self.averageFitness(), 4)) + " P:" + str(round(self.childSuccess, 4)))
        print("G:" + str(self.generation) + " CC:[" + ", ".join(str(m.categoryCoverage) for m in self.population) + "] AvgSig:" + str(round(self.avgSigma(), 3)) + " MemVar:" + str(round(self.memberVarience(), 4)) + " Alph:" + str(round(self.alphas[0].fitness, 4)) + " Avg: " + str(round(self.averageFitness(), 4)) + " P:" + str(round(self.childSuccess, 4)))



class GeneticAlgorithm(TrainingStrategy):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.strategy = TrainingStrategyType.GeneticAlgorithm
        self.childSuccess = 0.0
        self.highestCurrentMemberId = 0
        self.maxGenerations = 15
        self.childPopulation = []

    def select(self):
        self.population.sort(key=lambda x: x.fitness, reverse=False)
        if not self.alphas:
            self.alphas.append(self.population[0])
        else:
            self.alphas[0] = self.population[0]
        bestMembers = self.population[:len(self.population)/2]
        otherMembers = self.population[len(self.population)/2:]
        for i in range(len(bestMembers)):
            yield [bestMembers[i], otherMembers[i]]

    def crossover(self, parents):
        """For the """
        parents = list(parents)
        for j, gene in enumerate(parents[0]):
            child = Member(0, 1, self.useSigmas, self.sigmaMax)
            if j % 2 == 0:
                child.genome.append(parents[0][j])
            else:
                child.genome.append(parents[1][j])
            self.childPopulation.append(child)
        return self.childPopulation

    def mutate(self):
        for member in self.population:
            for i, gene in enumerate(member.genome):
                for j, elem in enumerate(gene):
                    if self.mutation():
                        if random.choice([True, False]):
                            member.genome[i][j] += self.epsilon()
                        else:
                            member.genome[i][j] -= self.epsilon()


    def mutation(self):
        # member = self.population[0]
        numberOfElements = len(Member.genomeTemplate)
        probability = 1 / float(numberOfElements)
        elem = random.uniform(0,numberOfElements) * probability
        choice = random.uniform(0, 1)
        diff = math.fabs(elem - choice)
        if diff > probability:
            return False
        return True

    def evaluateFitness(self, child):
        # fitness = 0
        # for pattern in patternSet.patterns:
        #     Network.Layer.setInputs(Network.Net[0], pattern['p'])
        #     Network.Layer.feedforward(Network.Net[0])
        #     fitness += Network.Net.calculateConvError(Network.Net, pattern['t'])
        # child.fitness = fitness
        return 0

    # def repopulate(self, contendors):
    #     bestFit = 0
    #     nextFit = 0
    #     for member in contendors:
    #         if member.fitness > bestFit:
    #             bestFitMember = member
    #         elif member.fitness > nextFit:
    #             nextFitMember = member
    #     self.population.append(bestFitMember)
    #     self.population.append(nextFitMember)
    def repopulate(self):
        self.repopulateGA()

    def repopulateGA(self):
        oldAndNew = self.population + self.childPopulation
        oldAndNew.sort(key=lambda x: x.fitness, reverse=False)
        self.population = oldAndNew[:self.maxPopulationSize]
        if not self.population:
            self.alphas.append(self.population[0])
        recordItems(", ".join(str(int(m.fitness)) for m in self.population) + ", " + str(self.childSuccess))
        self.childSuccess = 0.0
        for member in self.population:
            if member.id > self.highestCurrentMemberId:
                self.childSuccess = self.childSuccess + 1
        self.childSuccess = self.childSuccess/self.maxPopulationSize


class DifferentialGA(TrainingStrategy):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.strategy = TrainingStrategyType.DifferentialGA
        self.mask = []
        self.x = 'alpha' #way of selecting target: alpha, random
        self.y = 2        #number of difference vectors
        self.z = 'binomial' #crossover operator: mask, binomial. exponential
        self.beta = 0.2
        self.probability = 0.3
        self.mask
        self.useSigmas = False
        self.childSuccess = 0.0
        self.highestCurrentMemberId = 0
        self.maxGenerations = 15

    def select(self):
        return 0

    def selectTwo(self):
        return random.sample(self.population, 2)

    def createMask(self, target):
        mask = []
        for gene in target.genome:
            for elem in gene:
                prob = random.uniform(0, 1)
                if prob < self.probability:
                    mask.append(elem)
        self.mask = mask

    def crossover(self, parents):
        self.population.sort(key=lambda x: x.fitness, reverse=False)
        if not self.alphas:
            self.alphas.append(self.population[0])
        self.createMask(self.alphas[0])
        self.highestCurrentMemberId = Member.memberIdInc
        self.childPopulation = []
        for member in self.population:
            trial = self.mutateDiff()
            child = Member(0, 1, self.useSigmas, self.sigmaMax)
            for i, gene in enumerate(member.genome):
                for j, elem in enumerate(gene):
                    if elem not in self.mask:
                        child.genome[i][j] = elem
                    else:
                        child.genome[i][j] = trial.genome[i][j]
            self.childPopulation.append(child)
        return self.childPopulation

    def mutateDiff(self):
        target = self.alphas[0]
        randoms = self.selectTwo()
        random1 = randoms[0]
        random2 = randoms[-1]
        trial = Member(0,1,self.useSigmas, self.sigmaMax)
        for i, gene in enumerate(random1.genome):
            for j, elem in enumerate(gene):
                trial.genome[i][j] = target.genome[i][j] + (self.beta * (elem - random2.genome[i][j]))
        return trial

    def evaluateFitness(self):
        return 0

    def repopulate(self):
        self.repopulateDGA()

    def repopulateDGA(self):
        newPopulation = []
        for i, member in enumerate(self.population):
            if member.fitness > self.childPopulation[i]:
                newPopulation.append(member)
            else:
                newPopulation.append(self.childPopulation[i])
        newPopulation.sort(key=lambda x: x.fitness, reverse=False)
        self.population = newPopulation
        self.childSuccess = 0.0
        for member in self.population:
            if member.id > self.highestCurrentMemberId:
                self.childSuccess = self.childSuccess + 1
        self.childSuccess = self.childSuccess/self.maxPopulationSize
        #print("G:" + str(self.generation) + " CC:[" + ", ".join(str(m.categoryCoverage) + ":" + str(round(m.fitness, 4)) for m in self.population) + "] AvgSig:" + str(round(self.avgSigma(), 3)) + " MemVar:" + str(round(self.memberVarience(), 4)) + " Alph:" + str(round(self.alphas[0].fitness, 4)) + " Avg: " + str(round(self.averageFitness(), 4)) + " P:" + str(round(self.childSuccess, 4)))

    def mutate(self):
        return 0
