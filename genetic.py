from numpy import array, random

from neural import NeuralNetwork


class Individual:
    """ defines individual of a population """
    def __init__(self, layers=None, weights=None):
        self.score = 0
        # creating new random object
        if weights is None:
            self.nn = NeuralNetwork(layers=layers)
        # mutation
        else:
            self.nn = NeuralNetwork(layers=layers, weights=weights)
    
    def find_fitness(self):
        return self.score
    
    def reset(self):
        """ used during evolution of population """
        self.score = 0

class Population:
    """ defines functions related to population of genetic algorithm """
    def __init__(self, pop_size=1000, mutate_prob=0.03, retain_unfit_prob=0.01, select=0.333, layers=None):
        self.pop_size = pop_size                    # number of individuals consisiting population
        self.mutate_prob = mutate_prob              # probability that a gene is mutated        
        self.retain_unfit_prob = retain_unfit_prob  # propability of retaining unfit individuals
        self.select = select                        # fraction of fittest population being selected
        self.layers = layers                        # layers used in neural network
        self.fitness_history = []                   # stores fitness values for each generation

        self.generation = 1                         # holds current generation number
        # population initialization
        self.individuals = [Individual(layers=layers) for i in range(self.pop_size)]    
    
    def grade(self):
        """ finds population fitness """
        self.pop_fitness = max([i.find_fitness() for i in self.individuals])
        self.fitness_history.append(self.pop_fitness)
    
    def select_parents(self):
        """ selects fittest parents with few unfittest as well """
        self.individuals = sorted(self.individuals, key=lambda i: i.find_fitness(), reverse=True)
        # selecting fittest parents
        parents_selected = int(self.select * self.pop_size)
        self.parents = self.individuals[:parents_selected]
        # including some unfittest parents
        unfittest = self.individuals[parents_selected:]
        for i in unfittest:
            if self.retain_unfit_prob > random.rand():
                self.parents.append(i)
        
        # reset properties of parents
        for individual in self.parents:
            individual.reset()

    
    def crossover(self, weights1, weights2):
        """ combines the genes of two parent to form genes of child """
        weights = []

        for w1, w2 in zip(weights1, weights2):
            w = []
            for column1, column2 in zip(w1, w2):
                column = []
                for theta1, theta2 in zip(column1, column2):
                    # selecting randomly from father or mother genes
                    choosen = random.choice((theta1, theta2))       
                    column.append(choosen)
                w.append(column)
            weights.append(array(w))
        return weights

    def breed(self):
        """ creates new children for populating the population using fittest parents """
        children_size = self.pop_size - len(self.parents)
        children = []
        if len(self.parents) > 0:
            while len(children) < children_size:
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                if father != mother:
                    child_weights = self.crossover(father.nn.weights, mother.nn.weights)
                    child = Individual(layers=self.layers, weights=child_weights)
                    children.append(child)
            
            self.individuals = self.parents + children

    def evolve(self):
        """ define process of evolution """
        self.grade()
        self.select_parents()
        self.breed()

        print(f'{self.generation}  --> {self.fitness_history[-1]}') 
        self.generation += 1