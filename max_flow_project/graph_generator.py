"""
随机流量网络生成
输出 networkx有向图 + 容量邻接矩阵
源点 s=0；汇点 t = n_nodes‑1
"""
import networkx as nx
import numpy as np
from typing import Tuple


def generate_flow_graph(
    n_nodes: int,
    edge_density: float,
    cap_min: int,
    cap_max: int
) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    :return: G: DiGraph, cap_matrix: n×n容量邻接矩阵
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    cap_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    s = 0
    t = n_nodes - 1

    # 遍历所有i!=j候选边
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            # 随机按密度生成边
            if np.random.rand() < edge_density:
                cap = np.random.randint(cap_min, cap_max + 1)
                G.add_edge(i, j, capacity=cap)
                cap_matrix[i, j] = cap

    # 保证源点汇点至少存在一条路径，否则最大流恒0
    if not nx.has_path(G, s, t):
        cap = np.random.randint(cap_min, cap_max + 1)
        G.add_edge(s, t, capacity=cap)
        cap_matrix[s, t] = cap

    return G, cap_matrix


def generate_dataset(
    n_samples: int,
    n_nodes: int,
    edge_density: float,
    cap_min: int,
    cap_max: int,
    calc_maxflow_func
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量生成数据集
    calc_maxflow_func: 传入edmonds‑karp求解函数，用来打标签y_true
    return X:(samples, n_nodes*n_nodes), y:(samples,)
    """
    X_list = []
    y_list = []

    for idx in range(n_samples):
        G, cap_mat = generate_flow_graph(n_nodes, edge_density, cap_min, cap_max)
        max_flow_val, _ = calc_maxflow_func(G)
        feat = cap_mat.flatten()
        X_list.append(feat)
        y_list.append(max_flow_val)
        if (idx+1) % 500 == 0:
            print(f"[Dataset generate] {idx+1}/{n_samples} samples done")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y
