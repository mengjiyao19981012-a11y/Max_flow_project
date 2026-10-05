"""
single_compare.py
单张图对比：Edmonds‑Karp精确算法 VS MLP神经网络近似
任务书需求：对比精度、运行耗时
"""
import numpy as np
import torch
from typing import Dict
from config import DEVICE, TORCH_DTYPE
from edmonds_karp import edmonds_karp_max_flow
from nn_approximator import MaxFlowMLP
from utils import calc_metrics, timer


@timer
def nn_infer_single(model: MaxFlowMLP, cap_matrix: np.ndarray) -> float:
    """
    单张图神经网络推理
    :param model: 已加载训练好的MLP模型
    :param cap_matrix: n×n容量邻接矩阵
    :return:预测最大流浮点数
    """
    feat = cap_matrix.flatten()
    x_np = np.expand_dims(feat, axis=0)
    x_tensor = torch.from_numpy(x_np).to(device=DEVICE, dtype=TORCH_DTYPE)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor)
    return float(pred.cpu().numpy()[0])


def compare_single_graph(G, cap_matrix: np.ndarray, model: MaxFlowMLP) -> Dict:
    """
    对同一个流量图做两种算法完整对比
    返回字典：真值、预测、误差、两个算法耗时
    """
    # 精确算法
    true_flow, time_exact = edmonds_karp_max_flow(G)

    # NN推理
    pred_flow, time_nn = nn_infer_single(model, cap_matrix)

    # 计算误差
    abs_error = abs(true_flow - pred_flow)
    if true_flow != 0:
        rel_error = abs_error / true_flow
    else:
        rel_error = 0.0

    result = {
        "true_max_flow": int(true_flow),       # 转python原生int
        "pred_max_flow": float(round(pred_flow,4)),
        "abs_error": float(round(abs_error,4)),
        "rel_error": float(round(rel_error,4)),
        "time_exact_sec": float(round(time_exact,6)),
        "time_nn_sec": float(round(time_nn,6))
    }
    return result
