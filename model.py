import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import numpy as np


class EncodeChrom(nn.Module):
    def __init__(self, n_nodes):
        super(EncodeChrom, self).__init__()
        self.n_nodes = n_nodes

        self.conv1 = GATConv(9, 64)
        self.conv2 = GATConv(9, 64)
        self.lin1 = nn.Linear(64, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, data):
        x_1, x_2, edge_index_1, edge_index_2 = data.x_1, data.x_2, data.edge_index_1, data.edge_index_2
        out_1 = self.conv1(x_1, edge_index_1)
        out_2 = self.conv2(x_2, edge_index_2)
        out = out_1 + out_2
        out = out.view(data.num_graphs, self.n_nodes, -1)
        out = self.lin1(out)
        out = self.lin2(out)
        out = torch.squeeze(out, dim=-1)
        probs = F.softmax(out, dim=-1)
        distrib = Categorical(probs)

        return distrib


class Model:
    def __init__(self, config):
        super(Model, self).__init__()
        self.rollout_steps = config.rollout_steps
        self.n_nodes = config.n_nodes
        self.gamma = config.gamma
        self.device = config.device
        self.num_agents = config.n_agents
        self.train_batch_size = config.train_batch_size
        self.ppo_epoch = config.ppo_epoch
        self.encoder = EncodeChrom(self.n_nodes).to(config.device)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).float().to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards.unsqueeze_(1)

        dataset = []
        for i in range(len(memory.states)):
            chrom_1, chrom_2 = memory.states[i][0], memory.states[i][1]
            edge_index_n = torch.tensor(memory.edge_index_n[i], dtype=torch.long).to(self.device)
            action = torch.tensor(memory.actions[i], dtype=torch.int).to(self.device)
            action_log_prob = torch.tensor(memory.action_log_probs[i], dtype=torch.float32).to(self.device)
            edge_index_1 = torch.tensor(chrom_1.edge_index, dtype=torch.long).to(self.device)
            edge_index_2 = torch.tensor(chrom_2.edge_index, dtype=torch.long).to(self.device)
            x_1 = torch.tensor(chrom_1.embedding, dtype=torch.float32).to(self.device)
            x_2 = torch.tensor(chrom_2.embedding, dtype=torch.float32).to(self.device)
            data = Data(x_1=x_1,
                        x_2=x_2,
                        edge_index=edge_index_n,
                        edge_index_1=edge_index_1,
                        edge_index_2=edge_index_2,
                        action=action,
                        action_log_prob=action_log_prob)
            dataset.append(data)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

        for _ in range(self.ppo_epoch):
            for batch in loader:
                distrib = self.encoder(batch)


