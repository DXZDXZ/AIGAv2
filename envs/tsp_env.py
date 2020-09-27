import random
import copy
import numpy as np

from .utils import gen_coords, calc_disMat, calc_length


class TSP_Env:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.coords = gen_coords(n_nodes=self.n_nodes)
        self.disMat = calc_disMat(self.coords)
        self.pop_size = 2

        self.chroms = None
        self.state = None
        self.edge_index_n = None

    def reset(self):
        self.chroms = []
        for i in range(self.pop_size):
            seq = list(range(self.n_nodes))
            random.shuffle(seq)
            self.chroms.append(seq)

        # 图中任意两点皆有连边
        froms, tos = [], []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                froms.append(i)
                tos.append(j)
        self.edge_index_n = np.array([froms, tos])

        self.state = self.get_state()

    def get_cost(self):
        cost_1 = calc_length(self.chroms[0], self.disMat)
        cost_2 = calc_length(self.chroms[1], self.disMat)
        return (cost_1 + cost_2) / 2.0

    def get_state(self):
        state = {}

        # 两条染色体在同一个图上的连边
        froms, tos = [], []
        for i in range(self.n_nodes):
            if i == (self.n_nodes - 1):
                froms.extend([self.chroms[0][i], self.chroms[1][i], self.chroms[0][0], self.chroms[1][0]])
                tos.extend([self.chroms[0][0], self.chroms[1][0], self.chroms[0][i], self.chroms[1][i]])
            else:
                froms.extend([self.chroms[0][i], self.chroms[1][i], self.chroms[0][i+1], self.chroms[1][i+1]])
                tos.extend([self.chroms[0][i+1], self.chroms[1][i+1], self.chroms[0][i], self.chroms[1][i]])
        state['edge_index'] = np.array([froms, tos])

        # 每条染色体在图上分别的连边（这里是有向边），以及相应的边属性
        # TODO： 是否应该设置为无向边？ 或者设置为有向边，同时设置一个反向边也输入到模型中
        froms_1, tos_1, edge_attr_1 = [], [], []
        froms_2, tos_2, edge_attr_2 = [], [], []
        for i in range(self.n_nodes):
            if i == (self.n_nodes - 1):
                froms_1.append(self.chroms[0][i])
                tos_1.append(self.chroms[0][0])
                edge_attr_1.append([self.coords[self.chroms[0][i]][0],
                                    self.coords[self.chroms[0][i]][1],
                                    self.coords[self.chroms[0][0]][0],
                                    self.coords[self.chroms[0][0]][1],
                                    self.disMat[self.chroms[0][i]][self.chroms[0][0]]])
                froms_2.append(self.chroms[1][i])
                tos_2.append(self.chroms[1][0])
                edge_attr_2.append([self.coords[self.chroms[1][i]][0],
                                    self.coords[self.chroms[1][i]][1],
                                    self.coords[self.chroms[1][0]][0],
                                    self.coords[self.chroms[1][0]][1],
                                    self.disMat[self.chroms[1][i]][self.chroms[1][0]]])
            else:
                froms_1.append(self.chroms[0][i])
                tos_1.append(self.chroms[0][i+1])
                edge_attr_1.append([self.coords[self.chroms[0][i]][0],
                                    self.coords[self.chroms[0][i]][1],
                                    self.coords[self.chroms[0][i+1]][0],
                                    self.coords[self.chroms[0][i+1]][1],
                                    self.disMat[self.chroms[0][i]][self.chroms[0][i+1]]])
                froms_2.append(self.chroms[1][i])
                tos_2.append(self.chroms[1][i+1])
                edge_attr_2.append([self.coords[self.chroms[1][i]][0],
                                    self.coords[self.chroms[1][i]][1],
                                    self.coords[self.chroms[1][i+1]][0],
                                    self.coords[self.chroms[1][i+1]][1],
                                    self.disMat[self.chroms[1][i]][self.chroms[1][i+1]]])
        state['edge_index_1'] = np.array([froms_1, tos_1])
        state['edge_attr_1'] = np.array(edge_attr_1)
        state['edge_index_2'] = np.array([froms_2, tos_2])
        state['edge_attr_2'] = np.array(edge_attr_2)

        # Node Embedding（同时考虑两条染色体的情况）
        embeddings = []
        for cur_node in range(self.n_nodes):
            cur_node_index_1 = self.chroms[0].index(cur_node)
            cur_node_index_2 = self.chroms[1].index(cur_node)
            pre_node_1 = self.chroms[0][(cur_node_index_1 - 1 + self.n_nodes) % self.n_nodes]
            pre_node_2 = self.chroms[1][(cur_node_index_2 - 1 + self.n_nodes) % self.n_nodes]
            aft_node_1 = self.chroms[0][(cur_node_index_1 + 1) % self.n_nodes]
            aft_node_2 = self.chroms[1][(cur_node_index_2 + 1) % self.n_nodes]
            embedding = [self.coords[cur_node][0], self.coords[cur_node][1],
                         self.coords[pre_node_1][0], self.coords[pre_node_1][1],
                         self.coords[pre_node_2][0], self.coords[pre_node_2][1],
                         self.coords[aft_node_1][0], self.coords[aft_node_1][1],
                         self.coords[aft_node_2][0], self.coords[aft_node_2][1],
                         self.disMat[cur_node][pre_node_1],
                         self.disMat[cur_node][pre_node_2],
                         self.disMat[cur_node][aft_node_1],
                         self.disMat[cur_node][aft_node_2],
                         self.disMat[pre_node_1][aft_node_1],
                         self.disMat[pre_node_2][aft_node_2]]
            embeddings.append(embedding)
        state['embeddings'] = embeddings

        # Node Embedding（只考虑单独染色体的情况）
        embeddings_1, embeddings_2 = [], []
        for cur_node in range(self.n_nodes):
            cur_node_index_1 = self.chroms[0].index(cur_node)
            cur_node_index_2 = self.chroms[1].index(cur_node)
            pre_node_1 = self.chroms[0][(cur_node_index_1 - 1 + self.n_nodes) % self.n_nodes]
            pre_node_2 = self.chroms[1][(cur_node_index_2 - 1 + self.n_nodes) % self.n_nodes]
            aft_node_1 = self.chroms[0][(cur_node_index_1 + 1) % self.n_nodes]
            aft_node_2 = self.chroms[1][(cur_node_index_2 + 1) % self.n_nodes]
            embedding_1 = [self.coords[cur_node][0], self.coords[cur_node][1],
                           self.coords[pre_node_1][0], self.coords[pre_node_1][1],
                           self.coords[aft_node_1][0], self.coords[aft_node_1][1],
                           self.disMat[cur_node][pre_node_1],
                           self.disMat[cur_node][aft_node_1],
                           self.disMat[pre_node_1][aft_node_1]]
            embedding_2 = [self.coords[cur_node][0], self.coords[cur_node][1],
                           self.coords[pre_node_2][0], self.coords[pre_node_2][1],
                           self.coords[aft_node_2][0], self.coords[aft_node_2][1],
                           self.disMat[cur_node][pre_node_2],
                           self.disMat[cur_node][aft_node_2],
                           self.disMat[pre_node_2][aft_node_2]]
            embeddings_1.append(embedding_1)
            embeddings_2.append(embedding_2)
        state['embeddings_1'] = embeddings_1
        state['embeddings_2'] = embeddings_2

        return state

    def step(self, actions):
        actions = list(set(actions))
        n_actions = len(actions)
        max_step = int(self.n_nodes / n_actions)
        visited_nodes = []
        froms, tos = [], []
        for i in range(n_actions):
            action_node = actions[i]
            c_s = 0
            visited_nodes.append(action_node)
            while c_s < max_step and len(visited_nodes) != self.n_nodes:
                min_dis_node = np.argwhere(self.disMat[action_node] == sorted(self.disMat[action_node])[1]).item()
                ind = 2
                while (min_dis_node in visited_nodes or min_dis_node in actions) and ind < self.n_nodes:
                    min_dis_node = np.argwhere(self.disMat[action_node] == sorted(self.disMat[action_node])[ind]).item()
                    ind += 1
                if ind == self.n_nodes:
                    break
                if min_dis_node in visited_nodes:
                    break
                froms.append(action_node)
                tos.append(min_dis_node)

                visited_nodes.append(min_dis_node)
                action_node = min_dis_node
                c_s += 1
        edge_index = np.array([froms, tos])

    def step_2(self, actions):
        actions = list(set(actions))
        n_actions = len(actions)
        max_step = int(self.n_nodes / n_actions)
        visited_nodes = []
        froms, tos = [], []
        for i in range(n_actions):
            action_node = actions[i]
            edge_indexs = self.state['edge_index']
            nei_noods = []
            for j in range(edge_indexs[0]):
                if edge_indexs[0][j] == action_node:
                    nei_noods.append(edge_indexs[0][j])
                if edge_indexs[1][j] == action_node:
                    nei_noods.append(edge_indexs[1][j])

    def step_3(self, actions, length):
        cost_1 = calc_length(self.chroms[0], self.disMat)
        cost_2 = calc_length(self.chroms[1], self.disMat)

        f_action, m_action = actions[0], actions[1]
        f_nodes, m_nodes = [f_action], [m_action]
        f_action_index = self.chroms[0].index(f_action)
        m_action_index = self.chroms[1].index(m_action)
        # f_action_index, m_action_index = actions[0], actions[1]
        # f_action_node, m_action_node = self.chroms[0][f_action_index], self.chroms[1][m_action_index]
        # f_nodes, m_nodes = [f_action_node], [m_action_node]

        for i in range(1, int(length / 2) + 1):
            pre_f_action = self.chroms[0][(f_action_index - i + self.n_nodes) % self.n_nodes]
            pre_m_action = self.chroms[1][(m_action_index - i + self.n_nodes) % self.n_nodes]
            aft_f_action = self.chroms[0][(f_action_index + i) % self.n_nodes]
            aft_m_action = self.chroms[1][(m_action_index + i) % self.n_nodes]
            f_nodes.extend([pre_f_action, aft_f_action])
            m_nodes.extend([pre_m_action, aft_m_action])

        c_f_nodes, c_m_nodes = [], []
        for m_node, f_node in zip(self.chroms[1], self.chroms[0]):
            if m_node in f_nodes:
                c_f_nodes.append(m_node)
            if f_node in m_nodes:
                c_m_nodes.append(f_node)

        seq_1, seq_2 = [], []
        for i in range(1, self.n_nodes - length):
            seq_1.append(self.chroms[0][int(f_action_index + length / 2 + i) % self.n_nodes])
            seq_2.append(self.chroms[1][int(m_action_index + length / 2 + i) % self.n_nodes])
        seq_1.extend(c_f_nodes)
        seq_2.extend(c_m_nodes)

        new_cost_1 = calc_length(seq_1, self.disMat)
        new_cost_2 = calc_length(seq_2, self.disMat)
        self.chroms[0] = seq_1
        self.chroms[1] = seq_2
        self.state = self.get_state()

        # if new_cost_1 < cost_1 and new_cost_2 < cost_2:
        #     reward = 1
        # else:
        #     reward = 0
        reward = (cost_1 + cost_2) - (new_cost_1 + new_cost_2)

        return reward

    def step_4(self, action, length):
        pre_seq_1 = self.chroms[0]
        pre_seq_2 = self.chroms[1]
        pre_cost_1 = calc_length(self.chroms[0], self.disMat)       # 假设 chroms[0] 为主染色体
        pre_cost_2 = calc_length(self.chroms[1], self.disMat)       # chroms[1] 为辅染色体

        f_nodes = [action]
        f_action_index = self.chroms[0].index(f_nodes)

        for i in range(1, int(length / 2) + 1):
            pre_f_action = self.chroms[0][(f_action_index - i + self.n_nodes) % self.n_nodes]
            aft_f_action = self.chroms[0][(f_action_index + i) % self.n_nodes]
            f_nodes.extend([pre_f_action, aft_f_action])

        c_f_nodes = []
        for m_node in self.chroms[1]:
            if m_node in f_nodes:
                c_f_nodes.append(m_node)

        seq_1 = []
        for i in range(1, self.n_nodes - length):
            seq_1.append(self.chroms[0][int(f_action_index + length / 2 + i) % self.n_nodes])
        seq_1.extend(c_f_nodes)

        new_cost = calc_length(seq_1, self.disMat)

        self.chroms[0] = seq_1
        if pre_cost_1 < pre_cost_2:
            self.chroms[1] = pre_seq_1
        else:
            self.chroms[1] = pre_seq_2

        self.state = self.get_state()

        if new_cost < pre_cost_1:
            reward = 1
        else:
            reward = 0
        # reward = (cost_1 + cost_2) - (new_cost_1 + new_cost_2)

        return reward


if __name__ == '__main__':
    env = TSP_Env(10)
    env.reset()
    env.step_3([9, 3], 4)
