"""
Edmonds‑Karp 算法 (Ford‑Fulkerson BFS实现)
输入networkx有向流量图，返回最大流值
"""
import numpy as np
import networkx as nx
from collections import deque
from utils import timer


@timer
def edmonds_karp_max_flow(G: nx.DiGraph, s=0, t=None):
    """
    @timer装饰器：返回 (max_flow_value, cost_time)
    """
    if t is None:
        t = max(G.nodes)
    n = G.number_of_nodes()
    # 初始化残差图邻接矩阵
    residual = np.zeros((n, n), dtype=np.int32)
    for u, v, attr in G.edges(data=True):
        residual[u][v] = attr["capacity"]

    max_flow = 0

    def bfs():
        parent = [-1]*n
        q = deque([s])
        parent[s] = s
        while q:
            u = q.popleft()
            for v in range(n):
                if parent[v]==-1 and residual[u][v]>0:
                    parent[v]=u
                    q.append(v)
                    if v == t:
                        return parent
        return None

    while True:
        parent = bfs()
        if parent is None:
            break
        # 找增广路径最小残量
        v = t
        path_flow = float("inf")
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u
        v = t
        while v != s:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u
        max_flow += path_flow

    return max_flow
