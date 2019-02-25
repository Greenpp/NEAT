import random
import numpy as np

from matplotlib import pyplot as plt

from entity_original import Entity
from genetics_original import Gene
from specie_original import Specie


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

        # initial population
        first_entity = Entity(input_num, output_num)
        second_entity = Entity(input_num, output_num)
        self.insert_entity(first_entity)
        self.insert_entity(second_entity)

        Gene.innovation = 0

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

    def test_best(self, data, output, input_type='points', output_type='probability'):
        """
        Tests best network of population

        :param data: list
            List of vectors or which outputs will be printed
        :param output: list
            List of correct output vectors
        :param input_type: str, optional
            Network input type
        :param output_type: str, optional
            Network output type
        :return: bool
            Does best network match provided output
        """
        best_entity = self.species[0].entities[0]
        network = best_entity.create_network()

        net_out = network.compute(data, input_type, output_type)
        for y, y_hat in zip(net_out, output):
            y = np.round(y)

            if not np.array_equal(y, y_hat):
                return False

        return True

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
        offsprings = [int(round(s.shared_fitness / total_shared_fitness * self.population)) for s in old_species]

        # pass best entity of every specie unchanged
        # mate species to recreate population
        for specie, offspring in zip(old_species, offsprings):
            if offspring > 0:
                offspring -= 1
                self.species.append(specie.persist())
            mate_probabilities = list(reversed([i + 1 for i in range(len(specie.entities))]))
            for i in range(offspring):
                parent1, parent2 = random.choices(specie.entities, mate_probabilities, k=2)

                child = parent1.mate(parent2)
                self.insert_entity(child)

        # adapt acceptance delta
        # species = len(self.species)
        # if species < self.population // 4:
        #     self.specie_acceptance -= .1
        # else:
        #     self.specie_acceptance += .1

    def test(self, data, output, input_type='points', output_type='probability'):
        """
        Calculates fitness score for all entities

        :param data: list
            List of input vectors
        :param output: list
            List of expecting output vectors
        :param input_type: str, optional
            Network input type
        :param output_type: str, optional
            Network output type
        """
        for specie in self.species:
            specie.test(data, output, input_type, output_type)
