# create tsp class

import tsplib95
import matplotlib.pyplot as plt
import networkx as nx
import random
import math

class TSP:
    def __init__(self, cities, num_chromosomes, num_generations, mutation_rate, crossover_rate, verbose=False):
        self.cities = cities
        self.num_chromosomes = num_chromosomes
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.verbose = verbose
        # fitness is an array of num_chromosomes elements
        self.fitness = [0]*num_chromosomes
        self.elite = None

    def create_population(self):
        for _ in range(self.num_chromosomes):            
            chromosome = random.sample(range(1,len(self.cities.node_coords)+1), len(self.cities.node_coords))
            if self.verbose: 
                print(chromosome)
            self.population.append(chromosome)

    def distance(self, c1, c2):
        x1, y1 = self.cities.node_coords[c1]
        x2, y2 = self.cities.node_coords[c2]

        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    
    def evaluate(self):
        for idx, chromosome in enumerate(self.population):
            # calculate distance of chromosome
            distances = [self.distance(chromosome[i], chromosome[i-1]) for i in range(len(chromosome))]
            total_distance = sum(distances)
            self.fitness[idx] = 1 / total_distance

    def roulette_wheel(self):
        parents = random.choices(self.population, weights=self.fitness, k=2)
        # print
        if self.verbose:
            print("Parents selection: ")
            print(parents)
        return parents
    
    def rank_selection(self):
        # sort population by fitness
        population_sorted = [x for _, x in sorted(zip(self.fitness, self.population), key=lambda pair: pair[0])]
        # select parents, the higher the fitness, the higher the probability of being selected
        parents = random.choices(population_sorted, weights=list(range(1, len(population_sorted)+1)), k=2)
        return parents
    
    def tournament_selection(self):
        # select N/5 random chromosomes
        tournament = random.sample(self.population, self.num_chromosomes//5)
        # select the best 2
        parents = sorted(tournament, key=lambda x: self.fitness[self.population.index(x)], reverse=True)[:2]
        return parents
    
    def ordered_crossover(self, p1, p2):
        # Choose subset of cities
        subset_size = int(self.crossover_rate * len(p1))
        subset = set(random.sample(p1, subset_size))

        subset1 = [p for p in p1 if p in subset]
        subset2 = [p for p in p2 if p in subset]
        
        if self.verbose:
            print("Crossover: ")
            print(subset1)

        child1 = [subset2.pop(0) if p in subset else p for p in p1]
        child2 = [subset1.pop(0) if p in subset else p for p in p2]

        if self.verbose:
            print(child1)
            print(child2)

        return child1, child2
    
    def partially_mapped_crossover(self, father, mother):
        genes1 = father.copy()  # Initialize genes of child1 with father's genes
        genes2 = mother.copy()  # Initialize genes of child2 with mother's genes

        map1 = {gene: i for i, gene in enumerate(genes1)}  # Create a map for child1
        map2 = {gene: i for i, gene in enumerate(genes2)}  # Create a map for child2

        crossover_point1 = random.randint(1, len(father) - 2)  # Select 2 crossover points, excluding the first and last nodes
        crossover_point2 = random.randint(1, len(father) - 2)

        # Ensure crossover_point1 is smaller than crossover_point2
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1

        for i in range(crossover_point1, crossover_point2 + 1):
            value1, value2 = mother[i], father[i]

            # Swap genes in child1
            genes1[map1[value1]], genes1[i] = genes1[i], genes1[map1[value1]]
            # Swap indices in the map for child1
            map1[genes1[map1[value1]]], map1[genes1[i]] = map1[genes1[i]], map1[genes1[map1[value1]]]

            # Swap genes in child2
            genes2[map2[value2]], genes2[i] = genes2[i], genes2[map2[value2]]
            # Swap indices in the map for child2
            map2[genes2[map2[value2]]], map2[genes2[i]] = map2[genes2[i]], map2[genes2[map2[value2]]]

        child1 = genes1  # Assuming YourIndividualClass is the class representing an individual
        child2 = genes2
        return child1, child2
    
    def mutation(self, chromosome):
        if random.random() < self.mutation_rate:
            # chose a random city and put it before another city
            c1, c2 = random.sample(range(len(chromosome)), 2)
            chromosome.insert(c1, chromosome.pop(c2))
            if self.verbose:
                print("Mutation: ")
                print(chromosome)
        return chromosome
    
    def run(self):
        self.create_population()
        self.evaluate()
        print("Initial fitness: ")
        print(max(self.fitness))
        # choose max fitness chromosome as elite, which is a tuple (chromosome, fitness)
        self.elite = (self.population[self.fitness.index(max(self.fitness))], max(self.fitness))
        for i in range(self.num_generations):
            new_population = []
            for _ in range(self.num_chromosomes//2):
                # select parents
                p1, p2 = self.tournament_selection()
                # crossover
                c1, c2 = self.partially_mapped_crossover(p1, p2)
                # mutation
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
                # add to population
                new_population.append(c1)
                new_population.append(c2)
            # evaluate
            self.population = new_population
            self.population[-1] = self.elite[0]
            self.evaluate()
            # choose new elite if necessary
            if max(self.fitness) > self.elite[1]:
                self.elite = (self.population[self.fitness.index(max(self.fitness))], max(self.fitness))
            print("Fitness generation number " + str(i) + ": ")
            print(max(max(self.fitness), self.elite[1]))
        print("Final fitness: ")
        print(max(max(self.fitness), self.elite[1]))
        print("Elite: ")
        print(self.elite)
        # check if elite has repeated cities
        print("Elite has repeated cities: ")
        print(len(set(self.elite[0])) != len(self.elite[0]))



