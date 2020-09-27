import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes', default=20)
    parser.add_argument('--n_agents', default=64)
    parser.add_argument('--train_batch_size', default=64)
    parser.add_argument('--n_epoch', default=2000)
    parser.add_argument('--ppo_epochs', default=4)
    parser.add_argument('--n_anchors', default=2)
    parser.add_argument('--n_rollout', default=20)
    parser.add_argument('--rollout_steps', default=20)
    parser.add_argument('--eval_rollout_steps', default=1000)
    parser.add_argument('--lr', default=2e-7)
    parser.add_argument('--k_epoch', default=4)
    parser.add_argument('--eps_clip', default=0.2)
    parser.add_argument('--gamma', default=0.8)
    parser.add_argument('--betas', default=(0.9, 0.999))

    parser.add_argument('--node_dim', default=128)
    parser.add_argument('--embed_dim', default=16)
    parser.add_argument('--critic_h_dim', default=256)

    parser.add_argument('--device', default="cuda:0")
    return parser.parse_args()
