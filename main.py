# original - nodes without innovation number (transformed from connections)

import warnings

import numpy as np

from neat import NEAT

warnings.filterwarnings('ignore')

data = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]),
    np.array([0, 0])
]

output = [
    np.array([1]),
    np.array([1]),
    np.array([0]),
    np.array([0])
]

input_size = len(data[0])
output_size = len(output[0])

i_type = 'points'
o_type = 'probability'

x = NEAT(input_size, output_size)

x.test(data, output)
x.sort()
generation = 0
done = False
while True:
    generation += 1
    if generation % 10 == 0 and generation > 0:
        print('Generation {}'.format(generation))
    if generation % 100 == 0 and generation > 0:
        print('Best fitness: {}'.format(-x.species[0].get_best_fitness()))
        x.graph_best_network()
        x.graph_loss()
        x.info()
        x.show_best(data)
    if x.species[0].get_best_fitness() == 0:
        print('Finished on generation: {}'.format(generation))
        x.graph_best_network()
        x.graph_loss()
        break
    x.next_generation()
    x.test(data, output, i_type)
    x.sort()
