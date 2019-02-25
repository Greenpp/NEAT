from genetics import Gene


class Specie:
    """
    Specie of networks with similar genetic marker

    entities - list of specie entities
    marker - specie base genetic marker

    Marker is created from first entity in specie
    """

    def __init__(self, entity):
        """
        :param entity: Entity
            First specie entity
        """
        self.entities = [entity]
        self.shared_fitness = 0

    def sort(self):
        """
        Sorts entities by their fitness score
        """
        self.entities.sort(key=lambda e: e.fitness, reverse=True)

    def get_best_fitness(self):
        """
        Returns fitness of best entity in best specie

        :return: int
        """
        return self.entities[0].fitness

    def get_genetic_distance(self, entity, c1, c2, c3):
        """
        Calculates genetic distance between specie (best entity in specie) and given entity

        Distance is calculated using formula:
        c1 * Ex / N + c2 * Di / N + c3 * E(|dW|)

        c1, c2, c3 - weights
        Ex - number of excess genes
        Di - number of disjoint genes
        N - normalization factor, number of genes in bigger genetic pool
        E(|dW|) - average of matching genes weights differences absolute values

        :param c1: double
            Weight of excess genes
        :param c2: double
            Weight of disjoint genes
        :param c3:
            Weight of average weight difference
        :param entity: Entity
            Entity for which distance to specie will be calculated

        :return: double
            Distance value
        """

        # genes of both entities
        genes1 = self.entities[0].genotype.genes
        genes2 = entity.genotype.genes

        # full genetic information for historical markers
        genetics = Gene.genetics

        # historical markers for every gene
        markers_conns2 = {(next(iter(genetics[g])), g) for g in genes2}
        markers1 = {next(iter(genetics[g])) for g in genes1}
        markers2 = {t[0] for t in markers_conns2}

        # newest gene of specie entity, marks line between disjoint and excess genes
        if len(markers1) == 0:
            max_base_marker = 0
        else:
            max_base_marker = max(markers1)

        normalization_factor = max(len(markers1), len(markers2))
        if normalization_factor < 20:
            normalization_factor = 1

        # counters
        excess_num = 0
        disjoint_num = 0
        weight_dif = 0
        match_num = 0

        for m, c in markers_conns2:
            if m not in markers1:
                if m > max_base_marker:
                    # excess gene in entity 2
                    excess_num += 1
                else:
                    # disjoint gene in entity 2
                    disjoint_num += 1
            else:
                # matching gene
                match_num += 1
                w1 = genes1[c].weight
                w2 = genes2[c].weight

                diff = abs(w1 - w2)
                weight_dif += diff
        for m in markers1:
            if m not in markers2:
                # disjoint gene in entity 1
                disjoint_num += 1

        ex = excess_num
        di = disjoint_num
        we = weight_dif / match_num if match_num > 0 else 0
        n = normalization_factor

        return c1 * ex / n + c2 * di / n + c3 * we

    def test(self, data, output, input_type='points'):
        """
        Test every entity in specie

        :param data: list
            List of input vectors
        :param output: list
            List of output vectors
        :param input_type:
            Network input type
        """
        self.shared_fitness = 0
        specie_size = len(self.entities)

        for entity in self.entities:
            entity.test(data, output, input_type)

            self.shared_fitness += entity.fitness

        self.shared_fitness /= specie_size
