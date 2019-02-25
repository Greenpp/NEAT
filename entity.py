import random

from genetics import Genotype
from network import Network


class Entity:
    """
    Entity representing single neural network

    genotype - genotype encoding network structure
    fitness - fitness score of network
    """

    def __init__(self, input_num, output_num):
        """
        :param input_num: int
            Number of input layer nodes
        :param output_num:
            Number of output layer nodes
        """
        self.genotype = Genotype(input_num, output_num) if input_num > 0 and output_num > 0 else None
        self.fitness = None

    def create_network(self):
        """
        Decodes genotype to neural network

        :return: Network
        """
        return Network(self.genotype)

    def mate(self, entity):
        """
        Creates new entity using crossover function

        All genes are copied from more fit parent, then shared genes are replaced with second parent with 50% chance

        :param entity: Entity
            Second parent for mating
        :return: Entity
            Child entity created from crossover
        """
        # genotype1 is always from fitter entity
        genotype1 = self.genotype if self.fitness > entity.fitness else entity.genotype
        genotype2 = entity.genotype if self.fitness > entity.fitness else self.genotype

        # extracting matching genes
        matched_genes = set(genotype1.genes).intersection(set(genotype2.genes))

        child_gen = genotype1.copy()

        # genes replacement
        for mg in matched_genes:
            rand = random.random()
            if rand < .5:
                child_gen.genes[mg].weight = genotype2.genes[mg].weight

        # apply mutation
        child_gen.mutate()

        child = Entity(0, 0)
        child.genotype = child_gen

        return child

    def visualize(self):
        """
        Draws networkx plot of network
        """
        self.genotype.visualize()

    def test(self, data, output, input_type='points'):
        """
        Calculates entity fitness score base on given input vectors and expected outputs

        :param data: list
            List of input vectors
        :param input_type: str, optional
            Network input type
        :param output: list
            List of expected output vectors
        """
        self.fitness = 0
        network = Network(self.genotype)

        net_out = network.compute(data, input_type)
        for y, y_hat in zip(net_out, output):
            loss = (y - y_hat) ** 2
            self.fitness -= loss.item()

        self.fitness /= len(data)
