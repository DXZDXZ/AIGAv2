import random
import numpy as np
from GA.tools import gen_coords, calc_disMat
from GA.select import select_v1
from GA.crossover import crossover_v1
from GA.mutation import mutation_v1
from GA.update import update_v1


class TSPEnv:
    def __init__(self):
        self.pop_size = 50
        self.n_nodes = 20
        self.coords = gen_coords(self.n_nodes)
        self.disMat = calc_disMat(self.coords)

        self.chroms = []                                    # 初始化种群
        for i in range(self.pop_size):
            seq = list(range(self.n_nodes))
            random.shuffle(seq)
            self.chroms.append(seq)


if __name__ == '__main__':
    env = TSPEnv()

    for i_gen in range(10000):
        c_idx = select_v1(env)

        updated_chroms = None

        if np.random.randn() < 0.9:
            updated_chroms = crossover_v1(env, c_idx)

        if np.random.randn() < 0.1:
            updated_chroms = mutation_v1(env, c_idx)

        if updated_chroms is None:
            continue
        else:
            update_v1(env, updated_chroms)
