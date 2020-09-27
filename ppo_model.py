import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GATConv, MessagePassing, GlobalAttention, global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.data import Data, DataLoader
import numpy as np


class GatConvWithEdgeAttr(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2,dropout=0):
        super(GatConvWithEdgeAttr, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels+edge_channels,out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class CriticNet(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(CriticNet, self).__init__()
        self.lin_1 = nn.Linear(in_dim, h_dim)
        self.lin_2 = nn.Linear(h_dim, h_dim)
        self.lin_3 = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        x = self.lin_3(x)
        x = torch.sum(torch.squeeze(x, dim=2), dim=1)
        return x


class ActorNet(nn.Module):
    def __init__(self, embed_dim, node_dim):
        super(ActorNet, self).__init__()
        self.embed_dim = embed_dim
        self.node_dim = node_dim

        self.conv0 = GATConv(self.embed_dim, self.node_dim)
        # self.conv1_1 = GatConvWithEdgeAttr(self.node_dim, self.node_dim, 5)
        # self.conv1_2 = GatConvWithEdgeAttr(self.node_dim, self.node_dim, 5)
        self.conv1_1 = GATConv(self.node_dim, self.node_dim)
        self.conv1_2 = GATConv(self.node_dim, self.node_dim)
        self.lin1 = nn.Linear(self.node_dim, self.node_dim)
        self.lin2 = nn.Linear(self.node_dim, self.node_dim)
        self.lin3 = nn.Linear(self.node_dim, self.node_dim)
        # self.conv2 = GatConvWithEdgeAttr(self.node_dim, self.node_dim, 5)
        # self.conv3 = GatConvWithEdgeAttr(self.node_dim, self.node_dim, 5)
        self.conv2 = GATConv(self.node_dim, self.node_dim)
        self.conv3 = GATConv(self.node_dim, self.node_dim)
        self.conv4 = GATConv(self.node_dim, self.node_dim)
        self.lin4 = nn.Linear(self.node_dim, 1)

    def forward(self, data):
        x, x_1, x_2 = data.x, data.x_1, data.x_2
        edge_index_1, edge_index_2, edge_index_n = data.edge_index_1, data.edge_index_2, data.edge_index_n
        edge_attr_1, edge_attr_2 = data.edge_attr_1, data.edge_attr_2

        out = self.conv0(x, edge_index_n)
        # out_1 = self.conv1_1(out, edge_index_1, edge_attr_1)
        # out_2 = self.conv1_2(out, edge_index_2, edge_attr_2)
        out_1 = self.conv1_1(out, edge_index_1)
        out_2 = self.conv1_2(out, edge_index_2)
        # global_add_pool(out_1, batch=torch.from_numpy(np.arange(64).repeat(20)).to('cuda:0'))
        # out_cat = torch.cat([out_1, out_2], dim=-1)

        out_lin1 = self.lin1(out_1 + out_2)
        out_lin2 = self.lin2(out_lin1)
        out_lin3 = self.lin3(out_lin2) + out_lin1

        # out = self.conv2(out, edge_index_1, edge_attr_1)
        # out = self.conv3(out, edge_index_1, edge_attr_1)
        out_left = self.conv2(out_lin3, edge_index_1)
        out_left = self.conv3(out_left, edge_index_1)
        out_left = self.conv4(out_left, edge_index_n)
        out_left = out_left.view(data.num_graphs, int(data.num_nodes / data.num_graphs), -1)
        out_left = self.lin4(out_left)
        out_left = torch.squeeze(out_left, dim=-1)
        probs_left = F.softmax(out_left, dim=-1)
        distrib_left = Categorical(probs_left)

        out_right = self.conv2(out_lin3, edge_index_2)
        out_right = self.conv3(out_right, edge_index_2)
        out_right = self.conv4(out_right, edge_index_n)
        out_right = out_right.view(data.num_graphs, int(data.num_nodes / data.num_graphs), -1)
        out_right = self.lin4(out_right)
        out_right = torch.squeeze(out_right, dim=-1)
        probs_right = F.softmax(out_right, dim=-1)
        distrib_right = Categorical(probs_right)

        out_to_critic = out_lin3.view(data.num_graphs, int(data.num_nodes / data.num_graphs), -1)

        return distrib_left, distrib_right, out_to_critic


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.n_anchors = config.n_anchors
        self.device = config.device

        self.actor = ActorNet(config.embed_dim, config.node_dim)
        self.critic = CriticNet(config.node_dim, config.critic_h_dim, 1)

    def act(self, data):
        distrib_left, distrib_right, out = self.actor(data)
        action_node_left = distrib_left.sample((1,)).transpose(0, 1)
        action_node_right = distrib_right.sample((1,)).transpose(0, 1)
        return action_node_left, action_node_right, distrib_left, distrib_right

    def evaluate(self, memory_state, memory_nodes):
        distrib_left, distrib_right, out = self.actor(memory_state)

        action_log_probs = [distrib_left.log_prob(memory_nodes[:, 0]).cpu().detach().numpy(),
                            distrib_right.log_prob(memory_nodes[:, 1]).cpu().detach().numpy()]
        action_log_probs = torch.tensor(action_log_probs).transpose(0, 1).to(self.device)
        action_log_probs = torch.sum(action_log_probs, dim=1)

        state_value = self.critic(out)

        return action_log_probs, state_value


class Agent:
    def __init__(self, config):
        self.lr = config.lr
        self.betas = config.betas
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.ppo_epochs = config.ppo_epochs
        self.train_batch_size = config.train_batch_size
        self.device = config.device

        self.policy = ActorCritic(config).to(self.device)
        self.policy_old = ActorCritic(config).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MSELoss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = np.concatenate(rewards)
        rewards = torch.tensor(rewards).float().to(self.device)
        mean_rewards = rewards.mean().item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards.unsqueeze_(1)

        old_embeddings = torch.stack(memory.embeddings).to(self.device).detach()
        old_embeddings_1 = torch.stack(memory.embeddings_1).to(self.device).detach()
        old_embeddings_2 = torch.stack(memory.embeddings_2).to(self.device).detach()
        old_edge_index = torch.stack(memory.edge_index).to(self.device).detach()
        old_edge_index_n = torch.stack(memory.edge_index_n).to(self.device).detach()
        old_edge_index_1 = torch.stack(memory.edge_index_1).to(self.device).detach()
        old_edge_index_2 = torch.stack(memory.edge_index_2).to(self.device).detach()
        old_edge_attr_1 = torch.stack(memory.edge_attr_1).to(self.device).detach()
        old_edge_attr_2 = torch.stack(memory.edge_attr_2).to(self.device).detach()
        old_actions = torch.stack(memory.actions).unsqueeze_(1).to(self.device).detach()
        old_action_log_probs = torch.stack(memory.action_log_probs).unsqueeze_(1).to(self.device).detach()

        dataset = []
        for i in range(old_embeddings.size(0)):
            data = Data(x=old_embeddings[i],
                        x_1=old_embeddings_1[i],
                        x_2=old_embeddings_2[i],
                        edge_index=old_edge_index[i],
                        edge_index_n=old_edge_index_n[i],
                        edge_index_1=old_edge_index_1[i],
                        edge_index_2=old_edge_index_2[i],
                        edge_attr_1=old_edge_attr_1[i],
                        edge_attr_2=old_edge_attr_2[i],
                        actions=old_actions[i],
                        action_log_probs=old_action_log_probs[i],
                        rewards=rewards[i])
            dataset.append(data)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

        print_record = {}
        for ppo_epoch in range(self.ppo_epochs):
            for batch in loader:
                log_probs, state_value = self.policy.evaluate(batch, batch.actions)

                ratios = torch.exp(log_probs - batch.action_log_probs)
                advantages = batch.rewards - state_value.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = self.MSELoss(state_value, batch.rewards)
                loss = actor_loss + 0.5 * critic_loss

                print_record['actor_loss'] = actor_loss.mean().cpu().item()
                print_record['critic_loss'] = critic_loss.mean().cpu().item()
                print_record['loss'] = loss.mean().cpu().item()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            print("ppo_epoch %d/%d | actor_loss: %6f, critic_loss: %6f, total_loss: %6f" % (ppo_epoch,
                                                                                            self.ppo_epochs,
                                                                                            print_record['actor_loss'],
                                                                                            print_record['critic_loss'],
                                                                                            print_record['loss']))
        print_record['reward'] = mean_rewards
        self.policy_old.load_state_dict(self.policy.state_dict())

        return print_record

    def load(self, load_path):
        self.policy.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        self.policy_old.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    def save(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
