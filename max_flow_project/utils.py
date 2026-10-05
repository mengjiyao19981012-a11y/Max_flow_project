"""通用工具函数"""
import time
import json
import csv
import numpy as np
from pathlib import Path


def timer(func):
    """装饰器：统计函数执行时间，返回 (result, cost_seconds)"""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        cost = time.perf_counter() - t0
        return res, cost
    return wrapper


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    回归评估指标
    返回字典：mae, rmse, mean_relative_error
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(np.square(abs_err))))

    mask = y_true != 0
    if np.any(mask):
        rel_err = abs_err[mask] / y_true[mask]
        mean_rel_err = float(np.mean(rel_err))
    else:
        mean_rel_err = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mean_relative_error": mean_rel_err
    }


def save_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf‑8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_npz(filepath: Path, **arrays):
    np.savez_compressed(filepath, **arrays)


def write_csv(filepath: Path, header, rows):
    with open(filepath, "w", newline="", encoding="utf‑8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
