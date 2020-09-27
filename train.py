import torch
import copy
import time
import numpy as np
from torch_geometric.data import Data, DataLoader
from arguments import args
from memory import Memory
from ppo_model import Agent
from envs.tsp_env import TSP_Env

config = args()

if __name__ == '__main__':

    agent = Agent(config)

    for epoch in range(config.n_epoch):
        memory = Memory()
        envs = [TSP_Env(config.n_nodes) for i in range(config.n_agents)]

        time1 = time.time()
        next_value = []
        for i_rollout in range(config.n_rollout):
            # reset envs
            for i, env in enumerate(envs):
                env.reset()
            cost = np.array([env.get_cost() for env in envs]).mean()
            # if i_rollout % 5 == 0:
            #     print("i_rollout %2d/%d | init length: %f" % (i_rollout, config.n_rollout, cost))

            record_reward = []
            for i_step in range(config.rollout_steps):
                dataset = []
                for i, env in enumerate(envs):
                    # env = env.unwrapped
                    coords = env.coords
                    edge_index_n = env.edge_index_n
                    edge_index = env.state['edge_index']
                    embeddings = env.state['embeddings']
                    embeddings_1 = env.state['embeddings_1']
                    embeddings_2 = env.state['embeddings_2']
                    edge_index_1 = env.state['edge_index_1']
                    edge_index_2 = env.state['edge_index_2']
                    edge_attr_1 = env.state['edge_attr_1']
                    edge_attr_2 = env.state['edge_attr_2']

                    memory.embeddings.append(torch.tensor(copy.deepcopy(embeddings)).float().to(torch.device("cpu:0")))
                    memory.embeddings_1.append(
                        torch.tensor(copy.deepcopy(embeddings_1)).float().to(torch.device("cpu:0")))
                    memory.embeddings_2.append(
                        torch.tensor(copy.deepcopy(embeddings_2)).float().to(torch.device("cpu:0")))
                    memory.edge_index.append(torch.tensor(copy.deepcopy(edge_index)).long().to(torch.device("cpu:0")))
                    memory.edge_index_n.append(
                        torch.tensor(copy.deepcopy(edge_index_n)).long().to(torch.device("cpu:0")))
                    memory.edge_index_1.append(
                        torch.tensor(copy.deepcopy(edge_index_1)).long().to(torch.device("cpu:0")))
                    memory.edge_index_2.append(
                        torch.tensor(copy.deepcopy(edge_index_2)).long().to(torch.device("cpu:0")))
                    memory.edge_attr_1.append(
                        torch.tensor(copy.deepcopy(edge_attr_1)).float().to(torch.device("cpu:0")))
                    memory.edge_attr_2.append(
                        torch.tensor(copy.deepcopy(edge_attr_2)).float().to(torch.device("cpu:0")))

                    data = Data(x=torch.tensor(embeddings, dtype=torch.float32).to(config.device),
                                x_1=torch.tensor(embeddings_1, dtype=torch.float32).to(config.device),
                                x_2=torch.tensor(embeddings_2, dtype=torch.float32).to(config.device),
                                edge_index=torch.tensor(edge_index, dtype=torch.long).to(config.device),
                                edge_index_n=torch.tensor(edge_index_n, dtype=torch.long).to(config.device),
                                edge_index_1=torch.tensor(edge_index_1, dtype=torch.long).to(config.device),
                                edge_index_2=torch.tensor(edge_index_2, dtype=torch.long).to(config.device),
                                edge_attr_1=torch.tensor(edge_attr_1, dtype=torch.float32).to(config.device),
                                edge_attr_2=torch.tensor(edge_attr_2, dtype=torch.float32).to(config.device), )
                    dataset.append(data)
                loader = DataLoader(dataset, batch_size=config.n_agents, shuffle=False)

                action_node_left, action_node_right, distrib_left, distrib_right = agent.policy_old.act(list(loader)[0])

                action_log_probs = [distrib_left.log_prob(action_node_left[:, 0]).cpu().detach().numpy(),
                                    distrib_right.log_prob(action_node_right[:, 0]).cpu().detach().numpy()]
                action_log_probs = torch.tensor(action_log_probs).to(config.device).transpose(0, 1)
                action_log_probs = torch.sum(action_log_probs, dim=1)
                action_node_left = action_node_left.cpu().detach().numpy()
                action_node_right = action_node_right.cpu().detach().numpy()
                action_nodes = np.concatenate([action_node_left, action_node_right], 1)

                rewards = []
                for i, env in enumerate(envs):
                    # env = env.unwrapped
                    reward = env.step_3(action_nodes[i], 4)

                    memory.actions.append(torch.tensor(action_nodes[i]).to(torch.device("cpu:0")))
                    memory.action_log_probs.append(action_log_probs[i].to(torch.device("cpu:0")))
                    rewards.append(reward)
                record_reward.append(rewards)
                rewards = np.array(rewards)
                memory.rewards.append(rewards)
                memory.is_terminals.append(i_step == (config.rollout_steps - 1))

                # if i_rollout % 5 == 0 and i_step % 5 == 0:
                #     cost = np.array([env.get_cost() for env in envs]).mean()
                #     print(
                #         "iter %d/%d | mean rewards: %f | mean length: %f" % (i_step, config.rollout_steps, rewards.mean(), cost))
            with torch.no_grad:
                next_value = 0
            record_reward = np.array(record_reward)
            print("episode %d/%d | Mean/Median Reward: %5f/%5f, Min/Max Reward: %5f/%5f" %
                  (i_rollout, config.n_rollout, np.array(record_reward).mean(),
                   np.median(record_reward), np.min(record_reward),
                   np.max(record_reward)))

        if epoch % 10 == 0:
            for i_env, env in enumerate(envs):
                env.reset()
            for i_step in range(config.eval_rollout_steps):
                dataset = []
                for i, env in enumerate(envs):
                    # env = env.unwrapped
                    coords = env.coords
                    edge_index_n = env.edge_index_n
                    edge_index = env.state['edge_index']
                    embeddings = env.state['embeddings']
                    embeddings_1 = env.state['embeddings_1']
                    embeddings_2 = env.state['embeddings_2']
                    edge_index_1 = env.state['edge_index_1']
                    edge_index_2 = env.state['edge_index_2']
                    edge_attr_1 = env.state['edge_attr_1']
                    edge_attr_2 = env.state['edge_attr_2']

                    data = Data(x=torch.tensor(embeddings, dtype=torch.float32).to(config.device),
                                x_1=torch.tensor(embeddings_1, dtype=torch.float32).to(config.device),
                                x_2=torch.tensor(embeddings_2, dtype=torch.float32).to(config.device),
                                edge_index=torch.tensor(edge_index, dtype=torch.long).to(config.device),
                                edge_index_n=torch.tensor(edge_index_n, dtype=torch.long).to(config.device),
                                edge_index_1=torch.tensor(edge_index_1, dtype=torch.long).to(config.device),
                                edge_index_2=torch.tensor(edge_index_2, dtype=torch.long).to(config.device),
                                edge_attr_1=torch.tensor(edge_attr_1, dtype=torch.float32).to(config.device),
                                edge_attr_2=torch.tensor(edge_attr_2, dtype=torch.float32).to(config.device), )
                    dataset.append(data)
                loader = DataLoader(dataset, batch_size=config.n_agents, shuffle=False)

                with torch.no_grad():
                    action_node_left, action_node_right, distrib_left, distrib_right = agent.policy_old.act(
                        list(loader)[0])
                action_node_left = action_node_left.cpu().detach().numpy()
                action_node_right = action_node_right.cpu().detach().numpy()
                action_nodes = np.concatenate([action_node_left, action_node_right], 1)

                for i_env, env in enumerate(envs):
                    env.step_3(action_nodes[i_env], 4)

                if i_step % 50 == 0:
                    cost = np.array([env.get_cost() for env in envs]).mean()
                    print(
                        "eval iter %d/%d | mean length: %f" % (i_step, config.eval_rollout_steps, cost))

            agent.save("./trained_model/epoch_%d.pth" % epoch)

        print_record = agent.update(memory)
        print("Epoch %4d/%d | "
              "mean rewards: %5f | "
              "time consumed: %d min %ds" % (epoch,
                                             config.n_epoch,
                                             print_record['reward'],
                                             int((time.time() - time1) / 60),
                                             (time.time() - time1) % 60))

        memory.clear_memory()
