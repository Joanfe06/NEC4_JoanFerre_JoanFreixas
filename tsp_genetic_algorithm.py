# create tsp class

import tsplib95
import matplotlib.pyplot as plt
import networkx as nx
import random
import math

class TSP:
    def __init__(self, cities, num_chromosomes, num_generations, mutation_rate, crossover_rate):
        self.cities = cities
        self.num_chromosomes = num_chromosomes
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        # fitness is an array of num_chromosomes elements
        self.fitness = [0]*num_chromosomes

    def create_population(self):
        for _ in range(self.num_chromosomes):            
            chromosome = random.sample(range(len(self.cities.node_coords)), len(self.cities.node_coords))
            self.population.append(chromosome)

    def distance(self, c1, c2):
        return math.sqrt((self.cities.node_coords[c1][0]-self.cities.node_coords[c2][0])**2 + (self.cities.node_coords[c1][1]-self.cities.node_coords[c2][1])**2)
    
    def evaluate(self):
        for idx, chromosome in enumerate(self.population):
            # calculate distance of chromosome
            distances = [self.distance(chromosome[i]+1, chromosome[i+1]+1) for i in range(len(chromosome)-1)]
            distances.append(self.distance(chromosome[0]+1, chromosome[-1]+1))
            total_distance = sum(distances)
            self.fitness[idx] = 1 / total_distance if total_distance != 0 else float('inf')

    def select_parents(self):
        parents = random.choices(self.population, weights=self.fitness, k=2)
        return parents
    
    def mutation(self, chromosome):
        if random.random() < self.mutation_rate:
            # swap two random cities
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome
    
    def run(self):
        self.create_population()
        self.evaluate()
        print(self.fitness)
        for _ in range(self.num_generations):
            for _ in range(self.num_chromosomes/2):
                # select parents
                p1, p2 = self.select_parents()
                # crossover
                c1, c2 = self.crossover(p1, p2)
                # mutation
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)
                # add to population
                self.population.append(c1)
                self.population.append(c2)
            # evaluate
            self.evaluate()
            print(self.fitness)



