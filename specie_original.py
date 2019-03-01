from genetics_original import Gene


class Specie:
    """
    Specie of networks with similar genetic marker

    entities - list of specie entities
    marker - specie base genetic marker

    Marker is created from first entity in specie
    """

    def __init__(self, entity, age=0):
        """
        :param entity: Entity
            First specie entity
        :param age: int, optional
            Specie age
        """
        self.entities = [entity]
        self.shared_fitness = 0
        self.age = age

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
        markers1 = {genes1[conn].gene_id for conn in genes1}
        markers2 = {genes2[conn].gene_id for conn in genes2}

        # newest gene of specie entity, marks line between disjoint and excess genes
        if len(markers1) == 0:
            max_base_marker = 0
        else:
            max_base_marker = max(markers1)

        # normalization factor
        normalization_factor = max(len(markers1), len(markers2))
        if normalization_factor < 20:
            normalization_factor = 1

        # matching and not matching markers
        matching_markers = markers1.intersection(markers2)
        diff_markers = markers1.symmetric_difference(markers2)

        match_num = len(matching_markers)

        # weights difference
        total_weight_diff = 0
        for marker in matching_markers:
            connection = genetics[marker]

            w1 = genes1[connection].weight
            w2 = genes2[connection].weight

            weight_diff = abs(w1 - w2)
            total_weight_diff += weight_diff

        # disjoint and excess genes
        excess_num = 0
        disjoint_num = 0
        for marker in diff_markers:
            if marker <= max_base_marker:
                disjoint_num += 1
            else:
                excess_num += 1

        ex = excess_num
        di = disjoint_num
        # mean difference
        we = (total_weight_diff / match_num) if match_num > 0 else 0
        n = normalization_factor

        return c1 * ex / n + c2 * di / n + c3 * we

    def test(self, data, output, input_type='points', output_type='probability'):
        """
        Test every entity in specie

        :param data: list
            List of input vectors
        :param output: list
            List of output vectors
        :param input_type:
            Network input type
        :param output_type: str, optional
            Network output type
        """
        self.shared_fitness = 0
        specie_size = len(self.entities)

        for entity in self.entities:
            if entity.fitness is None:
                entity.test(data, output, input_type, output_type)

            self.shared_fitness += entity.fitness

        self.shared_fitness /= specie_size

    def persist(self):
        """
        Pushes specie to next generation

        :return: Specie
        """
        return Specie(self.entities[0], self.age + 1)
