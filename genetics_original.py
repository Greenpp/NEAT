import random

import networkx as nx
from matplotlib import pyplot as plt


class Gene:
    """
    Gene representing connection between nodes

    input_id - id of input node
    output_id - id of output node
    weight - weight of connection
    active - indicator if the gene is active

    genetics - static, maps gene id to input, output node tuple and other way around
    innovation - static, innovation number for marking new genes

    When the weight is not given, random between -2 and 2 will be picked
    """

    genetics = dict()
    innovation = 0

    def __init__(self, gene_id, weight=None, active=True):
        """
        :param gene_id: int
            Id of the gene, its innovation number
        :param weight: double, optional
            Weight of connection
        """
        self.gene_id = gene_id
        self.weight = weight if weight is not None else random.random() - .5
        self.active = active

    def copy(self):
        """
        Creates a copy of gene

        :return: Gene
        """
        cp = Gene(self.gene_id, self.weight, self.active)

        return cp

    def randomize(self):
        """
        Randomizes connection's weight
        """
        self.weight = random.random() * 4 - 2

    def random_shift(self):
        """
        Shifts weight by random value
        """
        self.weight += (random.random() - .5)


class Genotype:
    """
    Genotype encoding one network

    input_nodes - set of input layer nodes ids
    output_nodes - set of output layer nodes ids
    hidden_nodes - set of hidden layers nodes ids
    genes - dictionary of genes | (in,out): gene
    routes - dictionary of aviable connections for given node | id: [id1, id2, ...]
    """

    def __init__(self, input_num, output_num):
        """
        :param input_num: int
            Number of input layer nodes
        :param output_num: int
            Number of output layer nodes
        """
        self.input_nodes = set(range(input_num))
        self.output_nodes = set(range(input_num, input_num + output_num))
        self.hidden_nodes = set()
        self.genes = dict()
        self.routes = {b: {e for e in self.output_nodes} for b in self.input_nodes}

    def copy(self):
        """
        Creates a copy of genotype

        :return: Genotype
        """
        cp = Genotype(0, 0)

        cp.input_nodes = self.input_nodes.copy()
        cp.output_nodes = self.output_nodes.copy()
        cp.hidden_nodes = self.hidden_nodes.copy()
        cp.genes = {con: gen.copy() for con, gen in self.genes.items()}
        cp.routes = {n: s.copy() for n, s in self.routes.items()}

        return cp

    def insert_gene(self, connection, weight=None):
        """
        Creates new gen, inserts it into genome and manages global genetics

        :param connection: tuple
            Connected nodes (in, out)
        :param weight: double, optional
            Connection weight
        """
        # genetics - dictionary mapping id to connection (in,out) and connection to id
        genetics = Gene.genetics

        if connection in genetics:
            # connection between these nodes exists
            # innovation id is extracted from dictionary

            new_id = genetics[connection]
        else:
            # create new gene with innovation number
            new_id = Gene.innovation
            Gene.innovation += 1

            genetics[connection] = new_id
            genetics[new_id] = connection

        # remove output node from input node aviable connections
        in_node, out_node = connection
        self.routes[in_node].remove(out_node)

        # insert new connection gene
        gene = Gene(new_id, weight)
        self.genes[connection] = gene

    def add_connection(self):
        """
        Creates new connection in network
        """
        # pick input node
        in_nodes = [n for n in self.routes if len(self.routes[n]) > 0]
        if len(in_nodes) == 0:
            # network is fully connected
            return

        in_node = random.choice(in_nodes)

        # pick output node
        out_nodes = self.routes[in_node]
        out_node = random.choice(list(out_nodes))

        connection = (in_node, out_node)
        self.insert_gene(connection)

    def intersect_connection(self):
        """
        Splits existing connection with node, connection gene is disabled and replaced with two new

        Split gene is deleted and creates node with same id
        """
        # pick random connection
        connections = list(self.genes)
        if len(connections) == 0:
            # there are no connections
            return

        conn = random.choice(connections)

        gene = self.genes[conn]

        conn_weight = gene.weight

        # deactivate split connection
        # TODO remake as gene method
        gene.active = False

        # create two new connections
        in_node, out_node = conn
        new_node_id = len(self.input_nodes) + len(self.output_nodes) + len(self.hidden_nodes)

        new_in1 = in_node
        new_out1 = new_node_id

        new_in2 = new_node_id
        new_out2 = out_node

        connection1 = (new_in1, new_out1)
        connection2 = (new_in2, new_out2)

        # add new node to routes
        for key in self.routes:
            self.routes[key].add(new_node_id)
        # create routes for new node
        self.routes[new_node_id] = {h for h in self.hidden_nodes}
        self.routes[new_node_id].update({o for o in self.output_nodes})

        # add new genes
        self.insert_gene(connection1, weight=1)
        self.insert_gene(connection2, weight=conn_weight)

        # add new hidden node
        self.hidden_nodes.add(new_node_id)

    def mutate(self, weight_rate=.8, connection_rate=.05, node_rate=.03):
        """
        Mutates genotype with x probability for structural mutation and 1-x for weight mutation

        :param weight_rate: double, optional
            Probability of weights mutation
        :param connection_rate: double, optional
            Probability of adding new connection
        :param node_rate: double, optional
            Probability of splitting connection with new node
        """

        # weight mutation
        rand = random.random()
        if rand < weight_rate:
            for conn in self.genes:
                rand = random.random()
                if rand < .9:
                    # shift of weight
                    self.genes[conn].random_shift()
                else:
                    # randomization
                    self.genes[conn].randomize()

        # new connection mutation
        rand = random.random()
        if rand < connection_rate:
            self.add_connection()

        if len(self.genes) == 0:
            # no connections to split
            return

            # new node mutation
        rand = random.random()
        if rand < node_rate:
            self.intersect_connection()

    def visualize(self):
        """
        Creates networkx graph of encoded network
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.input_nodes)
        g.add_nodes_from(self.output_nodes)
        g.add_nodes_from(self.hidden_nodes)
        g.add_weighted_edges_from([(b, e, round(g.weight, 2)) for (b, e), g in self.genes.items() if g.active])

        pos = nx.shell_layout(g)
        labels = nx.get_edge_attributes(g, 'weight')
        nx.draw(g, pos=pos, with_labels=True)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
        plt.show()
