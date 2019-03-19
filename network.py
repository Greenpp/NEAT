import numpy as np


class Network:
    """
    Neural Network

    nodes - list of all network nodes
    input_nodes - list of input layer nodes, same as in nodes
    output_nodes - list of output layer nodes, same sa in nodes
    """

    def __init__(self, genotype):
        self.nodes = []
        self.input_nodes = []
        self.output_nodes = []

        # dictionary helps with id : node mapping
        nodes_dict = dict()
        # creation of input nodes without activation function
        for node_id in genotype.input_nodes:
            node = Node(activation='')
            self.input_nodes.append(node)
            self.nodes.append(node)
            nodes_dict[node_id] = node
        # creation of output nodes without activation function
        for node_id in genotype.output_nodes:
            node = Node(activation='')
            self.output_nodes.append(node)
            self.nodes.append(node)
            nodes_dict[node_id] = node
        # creation of hidden nodes with default activation function
        for node_id in genotype.hidden_nodes:
            node = Node(activation='sigmoid_custom')
            self.nodes.append(node)
            nodes_dict[node_id] = node

        # creating connections between nodes
        for conn in genotype.genes:
            weight = genotype.genes[conn].weight

            in_id, out_id = conn
            out_node = nodes_dict[out_id]
            in_node = nodes_dict[in_id]
            out_node.predecessors.append((in_node, weight))

    def compute(self, data, input_type='points', output_type='probability'):
        """
        Computes the forward pass of network for every input vector in data
        :param data: list
            List of input vectors
        :param input_type: str, optional
            Type of input data
                sequence will allow for recurrent connections
                points will test every input independent
        :param output_type: str, optional
            Type of output
                probability will return probability distribution of possible options
                raw will return raw network output
        :return: list
            List of output vectors
        """
        output = []

        for vector in data:
            # calculating output for every vector
            for v, node in zip(vector, self.input_nodes):
                node.set_value(v)

            result = [node.get_value() for node in self.output_nodes]

            result = np.array(result)
            if output_type == 'probability':
                if len(result) > 1:
                    # softmax calculation, multi class probability distribution
                    x = result
                    e_x = np.exp(x - np.max(x))

                    result = e_x / e_x.sum()
                else:
                    # sigmoid calculation, probability
                    x = result
                    x = np.clip(x, -500, 500)
                    result = 1 / (1 + np.exp(-x))

            output.append(result)

            # reset nodes
            for node in self.nodes:
                node.reset()
                if input_type == 'points':
                    node.past_value = 0

        return output


class Node:
    """
    Single node of neural network

    value - value of node
    activation - activation function of node
    visited - flag, describing if node was visited and value can be simply passed
    predecessors - list of tuples (node, weight) from which values are combined
    """

    def __init__(self, activation='sigmoid_custom'):
        self.value = 0
        self.past_value = 0
        self.visited = False
        self.calculated = False
        self.predecessors = []

        activations = {'ReLU': self.activation_ReLU,
                       'sigmoid_custom': self.activation_sigmoid_custom}
        self.activation = activations[activation] if activation in activations else None

    def activation_ReLU(self):
        """
        Executes ReLU function on node's value
        """
        self.value = max(0, self.value)

    def activation_sigmoid_custom(self):
        """
        Executes sigmoid function on node's value
        """
        self.value = 1 / (1 + np.e ** (-4.9 * self.value))

    def get_value(self):
        """
        Computes and returns value of the node, if node was visited previously and calculation isn't finished
        returns value of previous computation, acting like recurrent connection
        """
        if not self.visited:
            # first visit at node
            self.visited = True

            # value calculation
            for node, weight in self.predecessors:
                self.value += (node.get_value() * weight)

            # applying activation function
            if self.activation is not None:
                self.activation()

            self.calculated = True

            return self.value
        else:
            # visited node
            if self.calculated:
                # calculated in this computation
                return self.value
            else:
                # recurrent connection
                return self.past_value

    def set_value(self, new_val):
        """
        Sets node's value

        :param new_val: double
            New node's value
        """
        self.value = new_val

    def reset(self):
        """
        Resets node visit flag and value
        """
        self.visited = False
        self.calculated = False
        self.past_value = self.value
        self.value = 0
