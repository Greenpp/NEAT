import random

from matplotlib import pyplot as plt

from entity import Entity
from genetics import Gene
from specie import Specie


class NEAT:
    """
    NeuroEvolution of Augmented Topologies algorithm
    """

    def __init__(self, input_num, output_num, population=150, specie_acceptance=3, distance_params=(1, 1, .4)):
        """
        :param input_num: int
            Number of input layer nodes
        :param output_num: int
            Number of output layer nodes
        :param population: int
            Maximum size of population
        :param specie_acceptance: double, optional
            Acceptance percentage of specie marker size to its difference upon inserting new entity
        :param distance_params: tuple
            Weights for specie distance formula
        """
        self.input_num = input_num
        self.output_num = output_num
        self.population = population
        self.specie_acceptance = specie_acceptance
        self.distance_c1, self.distance_c2, self.distance_c3 = distance_params

        self.species = []

        self.error_value = []

        first_entity = Entity(input_num, output_num)
        second_entity = Entity(input_num, output_num)
        self.insert_entity(first_entity)
        self.insert_entity(second_entity)

        Gene.innovation = input_num + output_num

    def info(self):
        """
        Prints basic information about instance of NEAT (species number etc.)
        """
        # TODO add more information
        print('Species: {}'.format(len(self.species)))

    def graph_loss(self):
        """
        Creates plot with loss value over generations
        """
        # TODO change to graph and add  more information
        plt.plot(self.error_value)
        plt.title('Loss')
        plt.xlabel('Generation')
        plt.ylabel('Loss value')
        plt.show()

    def graph_best_network(self):
        """
        Graphs best network
        """
        self.species[0].entities[0].visualize()

    def insert_entity(self, entity, suggestion=None):
        """
        Inserts entity into one of species

        :param entity: Entity
            New entity
        :param suggestion: Specie, optional
            Suggested specie for new entity
        """
        if suggestion is not None:
            delta = suggestion.get_genetic_distance(entity, self.distance_c1, self.distance_c2, self.distance_c3)
            if delta < self.specie_acceptance:
                suggestion.entities.append(entity)
                return

        for specie in self.species:
            delta = specie.get_genetic_distance(entity, self.distance_c1, self.distance_c2, self.distance_c3)

            if delta < self.specie_acceptance:
                specie.entities.append(entity)
                return

        new_specie = Specie(entity)
        self.species.append(new_specie)

    def show_best(self, data):
        """
        # TODO remove ?
        Shows best network

        :param data: list
            List of vectors or which outputs will be printed
        """
        best_entity = self.species[0].entities[0]
        network = best_entity.create_network()

        print('Hidden units: {}'.format(len(best_entity.genotype.hidden_nodes)))

        outputs = network.compute(data)
        for i, o in zip(data, outputs):
            print('{} => {}'.format(i, o))

    def sort(self):
        """
        Sorts entities in species and species by the fitness value
        """
        for specie in self.species:
            specie.sort()

        self.species.sort(key=lambda s: s.get_best_fitness(), reverse=True)
        self.error_value.append(-self.species[0].get_best_fitness())

    def next_generation(self):
        """
        Creates next generation
        """
        # create new species pool and save old one
        old_species = self.species
        self.species = []

        # calculate number of offspring base on shared fitness
        total_shared_fitness = sum([s.shared_fitness for s in old_species])
        offsprings = [round(s.shared_fitness / total_shared_fitness * self.population) for s in old_species]

        # pass best entity unchanged as first specie child
        offsprings[0] -= 1
        self.insert_entity(old_species[0].entities[0])

        # mate species to recreate population
        for specie, offspring in zip(old_species, offsprings):
            mate_probabilities = list(reversed([i + 1 for i in range(len(specie.entities))]))
            for i in range(offspring):
                parent1, parent2 = random.choices(specie.entities, mate_probabilities, k=2)

                child = parent1.mate(parent2)
                self.insert_entity(child)

        # adapt acceptance delta
        species = len(self.species)
        if species < self.population // 4:
            self.specie_acceptance -= .1
        else:
            self.specie_acceptance += .1

    def test(self, data, output, input_type='points'):
        """
        Calculates fitness score for all entities

        :param data: list
            List of input vectors
        :param input_type: str, optional
            Network input type
        :param output: list
            List of expecting output vectors
        """
        for specie in self.species:
            specie.test(data, output, input_type)
