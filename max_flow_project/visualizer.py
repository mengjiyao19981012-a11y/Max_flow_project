"""
可视化工具：loss曲线绘制、流量网络图画图
只输出保存图片，不做GUI窗口
"""
import matplotlib
matplotlib.use("Agg")  # 无后台模式，只保存文件，macOS避免后端冲突
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path


def plot_loss_curve(csv_path: Path, save_png: Path):
    """读取train_log.csv，绘制train/test loss曲线保存图片"""
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(df["epoch"], df["train_loss"], label="Train loss", color="#1f77b4")
    plt.plot(df["epoch"], df["test_loss"], label="Test loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP training loss curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()
    print(f"Loss curve saved: {save_png}")


def draw_flow_network(G: nx.DiGraph, save_png: Path):
    """
    绘制流量有向网络图
    s=0(源点绿色), t=最大编号(汇点红色)
    边上显示容量
    """
    plt.figure(figsize=(9,7),dpi=150)
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    s = 0
    t = max(G.nodes)
    for node in G.nodes:
        if node == s:
            node_colors.append("#2ecc71")
        elif node == t:
            node_colors.append("#e74c3c")
        else:
            node_colors.append("#3498db")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=450)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, arrowstyle="->", alpha=0.7)

    edge_labels = {(u, v): attr["capacity"] for u, v, attr in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Flow network (source=green, sink=red)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()
    print(f"Network graph saved: {save_png}")
