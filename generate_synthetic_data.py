import numpy as np
import pandas as pd
import argparse
import os

from datetime import datetime, timedelta

################################### for generating synthetic data
from scipy.stats import lognorm,gamma
from scipy.optimize import brentq


def generate_hawkes1():
    [T, LL] = simulate_hawkes(100000, 0.05, [0.8, 0.0], [1.0, 20.0])
    score = - LL[80000:].mean()
    return [T, score]


def generate_hawkes2():
    [T, LL] = simulate_hawkes(100000, 0.1, [0.4, 0.4], [1.0, 20.0])
    score = - LL[80000:].mean()
    return [T, score]


def simulate_hawkes(n, mu, alpha, beta):
    T = []
    LL = []

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0

    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential() / l
        x = x + step

        l_trg_Int1 += l_trg1 * (1 - np.exp(-beta[0] * step)) / beta[0]
        l_trg_Int2 += l_trg2 * (1 - np.exp(-beta[1] * step)) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0] * step)
        l_trg2 *= np.exp(-beta[1] * step)
        l_next = mu + l_trg1 + l_trg2

        if np.random.rand() < l_next / l:  # accept
            T.append(x)
            LL.append(np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int)
            l_trg1 += alpha[0] * beta[0]
            l_trg2 += alpha[1] * beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1

            if count == n:
                break

    return [np.array(T), np.array(LL)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')

    parser.add_argument('--data_dir', type=str, default='./Synthetic')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--events_num', type=int, default=100000)
    parser.add_argument('--nodes_num', type=int, default=20)
    args = parser.parse_args()

    T_h, _ = simulate_hawkes(args.events_num, 0.15, [0.8, 0.0], [1.0, 20.0])

    rnd = np.random.RandomState(args.seed)
    node_pairs = []
    for i in range(args.nodes_num):
        for j in range(i + 1, args.nodes_num):
            node_pairs.append((i, j))
    node_pairs = np.array(node_pairs)

    pair_idx = rnd.choice(len(node_pairs), args.events_num, replace=True)

    syn_node_sequence = node_pairs[pair_idx]

    event_types = np.zeros(args.events_num, np.int)

    df_graph = pd.DataFrame({'event_types': event_types})

    df_graph['u'] = syn_node_sequence[:, 0]
    df_graph['i'] = syn_node_sequence[:, 1]

    ts = [0]
    for t in T_h[1:]:
        td = timedelta(hours=round(t - T_h[0], 2))
        t_cur = datetime.fromtimestamp(0) + td
        ts_cur = int(t_cur.timestamp())
        ts.append(ts_cur)

    df_graph['ts'] = ts

    df_graph.to_csv(os.path.join(args.data_dir, 'ml_hawkes.csv'), index=False)