"""
最大流神经网络近似 · iOS 风格桌面控制台

功能：
  1. 参数集中管理：点击功能区按钮 → 弹出该任务的参数调整面板（带推荐范围/校验）
     → 保存后自动关闭面板并开始执行任务
  2. 依赖串联：生成图 → 生成数据集 → 训练 → 对比，未满足前置条件时给出明确提示
  3. 进度显示：总进度条 + 阶段列表 + 已耗时/预计remaining + Epoch 实时面板
  4. 输出面板：训练/对比完成后可点击查看文字结果与图片
  5. 模型保存：训练完成后自动保存到项目根目录下的 models/ 文件夹
"""
from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
from matplotlib import font_manager

# macOS 上 PingFang SC / .ttc 往往不在 matplotlib 的字体索引里，直接注册后挑第一个可用的中文字体，
# 否则图表标题会变成方块。Windows / Linux 走各自的字体名兜底。
_CJK_FONT_CANDIDATES = (
    "PingFang SC", "PingFang HK", "Hiragino Sans GB", "Heiti TC", "Heiti SC",
    "STHeiti", "Arial Unicode MS", "Songti SC", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC",
)
_CJK_FONT_FILES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)


def _setup_cjk_font() -> list:
    for path in _CJK_FONT_FILES:
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return [name]
    return list(matplotlib.rcParams["font.family"])


matplotlib.rcParams["font.family"] = _setup_cjk_font()
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "#FFFFFF"
matplotlib.rcParams["axes.facecolor"] = "#FFFFFF"
matplotlib.rcParams["savefig.facecolor"] = "#FFFFFF"
matplotlib.rcParams["axes.edgecolor"] = "#D8D8DE"
matplotlib.rcParams["axes.labelcolor"] = "#3C3C43"
matplotlib.rcParams["xtick.color"] = "#8A8A8E"
matplotlib.rcParams["ytick.color"] = "#8A8A8E"
matplotlib.rcParams["text.color"] = "#1C1C1E"
matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import tkinter.filedialog as filedialog
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import BASE_DIR, DEVICE, OUTPUT_DIR
from edmonds_karp import edmonds_karp_max_flow
from graph_generator import generate_dataset, generate_flow_graph
from nn_approximator import MaxFlowMLP, build_dataloader, predict_mlp
from single_compare import compare_single_graph
from utils import calc_metrics, save_json, save_npz, write_csv

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------- 设计令牌(iOS)
IOS = dict(
    bg="#F2F2F7",
    card="#FFFFFF",
    card2="#F7F7FA",
    line="#E5E5EA",
    t1="#1C1C1E",
    t2="#3C3C43",
    t3="#8A8A8E",
    t4="#C7C7CC",
    blue="#007AFF",
    green="#34C759",
    indigo="#5856D6",
    orange="#FF9500",
    red="#FF3B30",
    teal="#30B0C7",
    pink="#FF2D55",
    purple="#AF52DE",
    gray="#F2F2F7",
)

FONT_FAMILY = "PingFang SC"
MONO_FAMILY = "Menlo"


def F(size: int, weight: str = "normal"):
    return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)


def FM(size: int, weight: str = "normal"):
    return ctk.CTkFont(family=MONO_FAMILY, size=size, weight=weight)


# CTkFont 在模块导入期无法创建（需要 Tk root），因此在 App 构造后再填充。
FONTS: Dict[str, Any] = {}


def ensure_fonts():
    """必须在 Tk root 创建之后调用。"""
    if FONTS:
        return FONTS
    FONTS.update(
        TITLE=F(17, "bold"), H1=F(21, "bold"), H2=F(15, "bold"), CARD=F(13, "bold"),
        BODY=F(13), SMALL=F(12), TINY=F(11), MONO=FM(11), BTN=F(13, "bold"),
        SMALL_B=F(12, "bold"), MONO_B=FM(11, "bold"),
    )
    globals().update({f"F_{k}": v for k, v in FONTS.items()})
    return FONTS

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
GRAPH_CACHE = OUTPUT_DIR / "gui_last_graph.npz"
DATASET_CACHE = OUTPUT_DIR / "gui_dataset.npz"
TRAIN_LOG_CSV = OUTPUT_DIR / "gui_train_log.csv"


def short_err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- 任务参数定义
def _spec(key, label, kind, default, lo=None, hi=None, options=None, tip="", affect="", unit=""):
    return dict(key=key, label=label, kind=kind, default=default, lo=lo, hi=hi,
                options=options, tip=tip, affect=affect, unit=unit)


GRAPH_PARAMS = [
    _spec("n_nodes", "顶点数量", "int", 8, 3, 30, tip="图里有多少个点。太小图没意义，太大会让后面的网"
                                                     "络输入维度暴涨。<b>推荐 6~14</b>，默认 8。",
          affect="决定 MLP 的输入维度 = 顶点数量²"),
    _spec("edge_density", "边密度", "float", 0.35, 0.05, 1.0,
          tip="任意两点之间连边的概率，取值 0~1。越密能流的路越多。<b>推荐 0.25~0.5</b>。",
          affect="影响最大流数值大小和求解难度"),
    _spec("cap_min", "边最小容量", "int", 1, 1, 100, tip="每条边容量的下界，必须 ≥1。一般保持 1 即可。",
          affect="决定最大流的最小可能值"),
    _spec("cap_max", "边最大容量", "int", 20, 2, 300,
          tip="每条边容量的上界，必须大于最小容量。<b>推荐 10~50</b>。",
          affect="决定最大流的最大可能值，越大预测越难"),
]

DATASET_PARAMS = [
    _spec("gen_new", "是否重新生成图参数", "bool", True,
          tip="开启后可顺便调整图的形态；关闭则完全沿用「生成网络图」时的参数。"),
    _spec("n_nodes", "顶点数量", "int", 8, 3, 30, tip="数据集中每张图的顶点数。<b>推荐 6~14</b>。",
          affect="样本特征维度 = 顶点数量²"),
    _spec("edge_density", "边密度", "float", 0.35, 0.05, 1.0, tip="0~1，<b>推荐 0.25~0.5</b>。"),
    _spec("cap_min", "边最小容量", "int", 1, 1, 100, tip="≥1。"),
    _spec("cap_max", "边最大容量", "int", 20, 2, 300, tip="<b>推荐 10~50</b>。"),
    _spec("train_samples", "训练样本数", "int", 6000, 100, 60000,
          tip="用于拟合的样本条数。每条都要跑一次精确算法，越多越慢。<b>推荐 3000~10000</b>。",
          affect="直接决定本项目约 80% 的耗时"),
    _spec("test_samples", "测试样本数", "int", 1500, 50, 30000,
          tip="用于评估泛化能力的样本条数，通常取训练集的 20%~30%。<b>推荐 1000~3000</b>。",
          affect="影响评估指标的可靠性"),
    _spec("use_cached", "优先使用已有数据集", "bool", False,
          tip="若之前已生成过数据集，勾选后可直接复用 <code>output/gui_dataset.npz</code>，跳过生成步骤。"),
]

TRAIN_PARAMS = [
    _spec("hidden_sizes", "隐藏层结构", "str", "128,64,32",
          tip="各隐藏层神经元个数，英文逗号分隔，例如 256,128,64。层数/宽度越大越容易过拟合。"
              "<b>推荐 3 层、逐层减半</b>。",
          affect="决定模型容量"),
    _spec("lr", "学习率 lr", "float", 1e-4, 1e-6, 1.0,
          tip="每次参数更新的步长。太大不收敛，太慢训练不动。Adam 优化器<b>推荐 1e-5 ~ 1e-3</b>。",
          affect="收敛速度与最终精度"),
    _spec("epochs", "训练轮数 epochs", "int", 80, 1, 3000,
          tip="把训练集完整过多少遍。<b>推荐 50~300</b>，配合曲线看是否收敛。",
          affect="训练总时长"),
    _spec("batch_size", "批大小 batch", "int", 128, 8, 2048,
          tip="每次更新用多少条样本。<b>推荐 64/128/256</b>，显存不足就调小。",
          affect="显存占用与梯度稳定性"),
    _spec("gen_new", "训练集参数", "bool", False, tip="勾选后可修改数据集参数，否则沿用上次生成的数据集。"),
    _spec("use_cached", "优先使用已有数据集", "bool", True,
          tip="若 output/gui_dataset.npz 存在且维度匹配就直接复用，能省下大量时间。"),
    _spec("train_samples", "训练样本数", "int", 6000, 100, 60000, tip="<b>推荐 3000~10000</b>。"),
    _spec("test_samples", "测试样本数", "int", 1500, 50, 30000, tip="<b>推荐 1000~3000</b>。"),
    _spec("n_nodes", "顶点数量", "int", 8, 3, 30, tip="<b>推荐 6~14</b>。"),
    _spec("edge_density", "边密度", "float", 0.35, 0.05, 1.0, tip="0~1。"),
    _spec("cap_min", "边最小容量", "int", 1, 1, 100, tip="≥1。"),
    _spec("cap_max", "边最大容量", "int", 20, 2, 300, tip="<b>推荐 10~50</b>。"),
    _spec("model_name", "模型保存文件名", "str", "maxflow_mlp",
          tip="保存到项目根目录 <code>models/</code> 下，自动补全时间戳与 .pth。"),
]

COMPARE_PARAMS = [
    _spec("source", "对比对象", "choice", "current", options=["current", "new"],
          tip="current = 使用当前已生成的那张图；new = 临时再随机生成一张新图来对比。"),
    _spec("n_nodes", "顶点数量", "int", 8, 3, 30, tip="仅在「新图」模式下生效，必须与模型输入维度一致。"),
    _spec("edge_density", "边密度", "float", 0.35, 0.05, 1.0, tip="仅在「新图」模式下生效。"),
    _spec("cap_min", "边最小容量", "int", 1, 1, 100, tip="≥1。"),
    _spec("cap_max", "边最大容量", "int", 20, 2, 300, tip="<b>推荐 10~50</b>。"),
]

TASK_DEF = {
    "graph": dict(name="生成网络图", icon="◎", color=IOS["teal"],
                  desc="随机生成一张带容量有向图，并用 Edmonds-Karp 求精确最大流",
                  params=GRAPH_PARAMS, run_key="gen_graph"),
    "dataset": dict(name="生成数据集", icon="▤", color=IOS["green"],
                    desc="批量生成随机图并计算标签，得到训练集 / 测试集",
                    params=DATASET_PARAMS, run_key="gen_dataset"),
    "train": dict(name="训练模型", icon="⌘", color=IOS["blue"],
                  desc="训练 MLP 拟合最大流，评估后保存到 models/ 目录",
                  params=TRAIN_PARAMS, run_key="train"),
    "compare": dict(name="算法对比", icon="⇄", color=IOS["indigo"],
                    desc="同一张图上对比 精确算法 与 神经网络 的结果与耗时",
                    params=COMPARE_PARAMS, run_key="compare"),
}

# 图/数据集通用参数键 -> 用于任务间参数联动
GRAPH_KEYS = ("n_nodes", "edge_density", "cap_min", "cap_max")

# 侧边栏卡片上的参数摘要（一眼看出当前配置）
def _graph_summary(p):
    try:
        return f"{int(p['n_nodes'])} 顶点 · 密度 {float(p['edge_density']):g} · 容量 {int(p['cap_min'])}~{int(p['cap_max'])}"
    except Exception:
        return "参数待设置"


def _dataset_summary(p):
    try:
        return f"训练 {int(p['train_samples'])} · 测试 {int(p['test_samples'])} · {int(p['n_nodes'])} 顶点"
    except Exception:
        return "参数待设置"


def _train_summary(p):
    try:
        lr = float(p["lr"])
        lr_s = f"{lr:.0e}".replace("e-0", "e-")
        return f"{p['hidden_sizes']} · {int(p['epochs'])} 轮 · lr {lr_s} · batch {int(p['batch_size'])}"
    except Exception:
        return "参数待设置"


def _compare_summary(p):
    return "使用当前图" if str(p.get("source", "current")) == "current" else "临时生成新图"


TASK_CHIPS = dict(graph=_graph_summary, dataset=_dataset_summary,
                  train=_train_summary, compare=_compare_summary)

# ---------------------------------------------------------------- 图表配色
CHART = dict(
    paper="#FFFFFF",
    panel="#FBFCFE",
    ink="#1C1C1E",
    ink2="#5A5A63",
    ink3="#9A9AA2",
    grid="#EDEDF2",
    blue="#007AFF",
    cyan="#32ADE6",
    green="#34C759",
    indigo="#5856D6",
    purple="#AF52DE",
    pink="#FF2D55",
    orange="#FF9500",
    red="#FF3B30",
    teal="#30B0C7",
)

# 渐变色带
CMAP_MAIN = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ios_main", ["#32ADE6", "#007AFF", "#5856D6", "#AF52DE"])
CMAP_WARM = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ios_warm", ["#FFD60A", "#FF9500", "#FF375F"])
CMAP_COOL = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ios_cool", ["#A8E8D8", "#30B0C7", "#007AFF"])


def render_tip(text: str) -> str:
    """把 <b>/<code> 标记转成普通文本（文本框不支持富文本）。"""
    return text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")


class ParamDialog(ctk.CTkToplevel):
    """iOS 风格参数调整面板：带推荐范围、校验、保存后自动关闭。"""

    def __init__(self, master, task_key: str, values: Dict[str, Any], on_save):
        super().__init__(master)
        self.task_key = task_key
        self.defn = TASK_DEF[task_key]
        self.on_save = on_save
        self.values: Dict[str, Any] = dict(values or {})
        self.result: Optional[Dict[str, Any]] = None
        self.vars: Dict[str, Any] = {}
        self.widgets: Dict[str, Any] = {}
        self.err_labels: Dict[str, Any] = {}
        self.choice_map: Dict[str, Dict[str, Any]] = {}

        accent = self.defn["color"]
        self.title("")
        self.configure(fg_color=IOS["bg"])
        self.transient(master)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._build_header(accent)
        self._build_body(accent)
        self._build_footer(accent)

        self.update_idletasks()
        w, h = 560, min(760, 190 + len(self.defn["params"]) * 84)
        self.geometry(f"{w}x{h}")
        self._center(w, h)
        self.lift()
        self.focus_force()
        self.grab_set()

    def _center(self, w, h):
        self.update_idletasks()
        mx, my = self.master.winfo_x(), self.master.winfo_y()
        mw, mh = self.master.winfo_width(), self.master.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        if mw > 10 and mh > 10:
            x = mx + (mw - w) // 2
            y = my + (mh - h) // 2
        else:
            x, y = (sw - w) // 2, (sh - h) // 2
        x = max(0, min(x, sw - w))
        y = max(0, min(y, sh - h - 40))
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_header(self, accent):
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=22, pady=(20, 8))
        icon = ctk.CTkFrame(head, width=44, height=44, corner_radius=22, fg_color=accent)
        icon.pack(side="left", padx=(0, 12))
        icon.pack_propagate(False)
        ctk.CTkLabel(icon, text=self.defn["icon"], font=F(20, "bold"),
                     text_color="#FFFFFF").pack(expand=True)
        box = ctk.CTkFrame(head, fg_color="transparent")
        box.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(box, text=self.defn["name"], font=F_H2, text_color=IOS["t1"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(box, text="调整本次运行的参数，保存后立即开始执行", font=F_TINY,
                     text_color=IOS["t3"], anchor="w").pack(anchor="w", pady=(2, 0))

    def _build_body(self, accent):
        body = ctk.CTkScrollableFrame(self, fg_color="transparent",
                                      scrollbar_button_color=IOS["t4"],
                                      scrollbar_button_hover_color=IOS["t3"])
        body.pack(fill="both", expand=True, padx=16, pady=(4, 4))
        body.grid_columnconfigure(0, weight=1)

        group_card = ctk.CTkFrame(body, fg_color=IOS["card"], corner_radius=14,
                                  border_width=1, border_color=IOS["line"])
        group_card.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(group_card, text="参数", font=F_CARD, text_color=IOS["t1"]).pack(
            anchor="w", padx=16, pady=(14, 4))

        for spec in self.defn["params"]:
            row = ctk.CTkFrame(group_card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=7)
            self._build_row(row, spec, accent)

    def _build_row(self, row, spec, accent):
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=spec["label"], font=F_BODY, text_color=IOS["t1"],
                     anchor="w").pack(side="left")
        val = self._initial_value(spec)
        err = ctk.CTkLabel(top, text="", font=F_TINY, text_color=IOS["red"], anchor="e")
        err.pack(side="right")
        self.err_labels[spec["key"]] = err

        if spec["kind"] == "bool":
            var = ctk.BooleanVar(value=bool(val))
            self.vars[spec["key"]] = var
            sw = ctk.CTkSwitch(row, text="", variable=var, progress_color=accent,
                               button_color="#FFFFFF", button_hover_color="#F0F0F5",
                               width=46, height=26, switch_width=46, switch_height=26)
            sw.pack(anchor="w", pady=(4, 0))
            self.widgets[spec["key"]] = sw
            self._hint(row, render_tip(spec["tip"]))
            return

        if spec["kind"] == "choice":
            labels = {"current": "使用当前图", "new": "临时生成新图"}
            opts = [labels.get(o, str(o)) for o in spec["options"]]
            real_by_label = dict(zip(opts, spec["options"]))
            label_by_real = {str(v): k for k, v in real_by_label.items()}
            var = ctk.StringVar(value=label_by_real.get(str(val), opts[0]))
            self.vars[spec["key"]] = var
            self.choice_map[spec["key"]] = real_by_label
            seg = ctk.CTkSegmentedButton(row, values=opts, variable=var, font=F_SMALL,
                                         height=32, corner_radius=9,
                                         fg_color=IOS["gray"], selected_color=IOS["card"],
                                         selected_hover_color=IOS["card"],
                                         unselected_color=IOS["gray"],
                                         unselected_hover_color="#E8E8ED",
                                         text_color=IOS["t2"])
            seg.pack(anchor="w", pady=(4, 0))
            self.widgets[spec["key"]] = seg
            self._hint(row, render_tip(spec["tip"]))
            return

        # int / float / str -> Entry
        var = ctk.StringVar(value=str(val))
        self.vars[spec["key"]] = var
        ent = ctk.CTkEntry(row, textvariable=var, font=F_BODY, height=36, corner_radius=10,
                           border_width=1, border_color=IOS["line"], fg_color=IOS["card2"],
                           text_color=IOS["t1"])
        ent.pack(fill="x", pady=(4, 0))
        self.widgets[spec["key"]] = ent

        rng = ""
        if spec["kind"] in ("int", "float") and spec["lo"] is not None:
            rng = f"范围 {spec['lo']} ~ {spec['hi']}"
        tips = [t for t in (rng, render_tip(spec["tip"])) if t]
        self._hint(row, " · ".join(tips) if rng else render_tip(spec["tip"]))
        if spec.get("affect"):
            ctk.CTkLabel(row, text=f"影响：{spec['affect']}", font=F_TINY,
                         text_color=IOS["t4"], anchor="w",
                         justify="left").pack(anchor="w", pady=(2, 0))

    def _hint(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=F_TINY, text_color=IOS["t3"], anchor="w",
                     wraplength=470, justify="left").pack(anchor="w", pady=(3, 0))

    def _initial_value(self, spec):
        """面板里的初始值：优先用外部传入的当前值，否则用推荐默认值。"""
        if spec["key"] in self.values:
            return self.values[spec["key"]]
        return spec["default"]

    def _build_footer(self, accent):
        foot = ctk.CTkFrame(self, fg_color=IOS["card"], corner_radius=0, height=76)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        ctk.CTkFrame(foot, height=1, fg_color=IOS["line"]).pack(fill="x")
        inner = ctk.CTkFrame(foot, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=14)
        ctk.CTkButton(inner, text="取消", font=F_BTN, height=42, corner_radius=12,
                      fg_color=IOS["gray"], text_color=IOS["t2"], hover_color="#E5E5EA",
                      command=self._cancel).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(inner, text="保存并运行", font=F_BTN, height=42, corner_radius=12,
                      fg_color=accent, text_color="#FFFFFF", hover_color=accent,
                      command=self._save).pack(side="left", fill="x", expand=True, padx=(6, 0))

    def _validate(self) -> bool:
        ok = True
        for spec in self.defn["params"]:
            key, kind = spec["key"], spec["kind"]
            err = self.err_labels[key]
            err.configure(text="")
            if kind == "bool":
                continue
            if kind == "choice":
                continue
            raw = str(self.vars[key].get()).strip()
            if kind == "str":
                if key == "hidden_sizes":
                    try:
                        vals = [int(x) for x in raw.split(",") if x.strip()]
                        assert vals and all(v > 0 for v in vals)
                    except Exception:
                        err.configure(text="格式应为 128,64,32")
                        ok = False
                elif not raw:
                    err.configure(text="不能为空")
                    ok = False
                continue
            try:
                num = float(raw)
            except ValueError:
                err.configure(text="请输入数字")
                ok = False
                continue
            if kind == "int" and abs(num - int(num)) > 1e-9:
                err.configure(text="请输入整数")
                ok = False
                continue
            num = int(num) if kind == "int" else num
            lo, hi = spec.get("lo"), spec.get("hi")
            if lo is not None and num < lo:
                err.configure(text=f"不能小于 {lo}")
                ok = False
            elif hi is not None and num > hi:
                err.configure(text=f"不能大于 {hi}")
                ok = False
        return ok

    def _collect(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for spec in self.defn["params"]:
            key, kind = spec["key"], spec["kind"]
            if kind == "bool":
                out[key] = bool(self.vars[key].get())
            elif kind == "choice":
                out[key] = self.choice_map[key].get(str(self.vars[key].get()),
                                                    spec["default"])
            elif kind == "str":
                out[key] = str(self.vars[key].get()).strip()
            else:
                num = float(str(self.vars[key].get()).strip())
                out[key] = int(num) if kind == "int" else num
        return out

    def _save(self):
        if not self._validate():
            return
        self.result = self._collect()
        cb = self.on_save
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
        if cb:
            cb(self.result)

    def _cancel(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class OutputDialog(ctk.CTkToplevel):
    """查看任务输出：文字结果 + 图片。"""

    def __init__(self, master, title: str, lines: List[str], images: List[str], accent=IOS["blue"]):
        super().__init__(master)
        self.title("")
        self.configure(fg_color=IOS["bg"])
        self.transient(master)
        self.images = [p for p in images if p and Path(p).exists()]
        self.lines = lines

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1080, int(sw * 0.78)), min(820, int(sh * 0.82))
        self.geometry(f"{w}x{h}")
        self._center(w, h)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(20, 10))
        ctk.CTkLabel(head, text=title, font=F_H2, text_color=IOS["t1"], anchor="w").pack(anchor="w")
        ctk.CTkLabel(head, text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=F_TINY,
                     text_color=IOS["t3"], anchor="w").pack(anchor="w", pady=(2, 0))

        tabs = ctk.CTkTabview(self, fg_color=IOS["bg"], corner_radius=12,
                              segmented_button_fg_color=IOS["gray"],
                              segmented_button_selected_color=IOS["card"],
                              segmented_button_selected_hover_color=IOS["card"],
                              segmented_button_unselected_color=IOS["gray"],
                              segmented_button_unselected_hover_color="#E8E8ED",
                              text_color=IOS["t2"], border_width=0)
        tabs.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        tabs.add("文字结果")
        if self.images:
            tabs.add(f"图片 ({len(self.images)})")

        txt = ctk.CTkTextbox(tabs.tab("文字结果"), font=F_MONO, fg_color=IOS["card"],
                             border_width=1, border_color=IOS["line"], corner_radius=12,
                             text_color=IOS["t1"])
        txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", "\n".join(self.lines))
        txt.configure(state="disabled")

        if self.images:
            scroll = ctk.CTkScrollableFrame(tabs.tab(f"图片 ({len(self.images)})"),
                                            fg_color="transparent",
                                            scrollbar_button_color=IOS["t4"],
                                            scrollbar_button_hover_color=IOS["t3"])
            scroll.pack(fill="both", expand=True)
            scroll.grid_columnconfigure(0, weight=1)
            for idx, path in enumerate(self.images):
                self._render_image(scroll, path, idx)

        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.pack(fill="x", padx=24, pady=(0, 18))
        if self.images:
            ctk.CTkButton(foot, text="导出图片到…", font=F_BTN, height=38, corner_radius=11,
                          fg_color=IOS["gray"], text_color=IOS["t2"], hover_color="#E5E5EA",
                          command=self._export).pack(side="left")
        ctk.CTkButton(foot, text="关闭", font=F_BTN, height=38, corner_radius=11,
                      width=110, fg_color=accent, text_color="#FFFFFF",
                      command=self.destroy).pack(side="right")
        self.lift()
        self.focus_force()

    def _center(self, w, h):
        mx, my = self.master.winfo_x(), self.master.winfo_y()
        mw, mh = self.master.winfo_width(), self.master.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        if mw > 10 and mh > 10:
            x, y = mx + (mw - w) // 2, my + (mh - h) // 2
        else:
            x, y = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y - 30)}")

    def _render_image(self, parent, path, idx):
        try:
            from PIL import Image
            im = Image.open(path)
            max_w = 720
            ratio = max_w / im.width
            size = (max_w, int(im.height * ratio)) if im.width > max_w else (im.width, im.height)
            photo = ctk.CTkImage(light_image=im.copy(), dark_image=im.copy(), size=size)
            im.close()
            card = ctk.CTkFrame(parent, fg_color=IOS["card"], corner_radius=12,
                                border_width=1, border_color=IOS["line"])
            card.grid(row=idx, column=0, sticky="ew", pady=6)
            ctk.CTkLabel(card, text=Path(path).name, font=F_TINY, text_color=IOS["t3"],
                         anchor="w").pack(anchor="w", padx=14, pady=(10, 4))
            lbl = ctk.CTkLabel(card, text="", image=photo)
            lbl.image = photo
            lbl.pack(padx=14, pady=(0, 14))
        except Exception as e:
            ctk.CTkLabel(parent, text=f"{Path(path).name} 预览失败：{short_err(e)}",
                         font=F_SMALL, text_color=IOS["red"]).grid(row=idx, column=0, pady=6)

    def _export(self):
        if len(self.images) == 1:
            dest = filedialog.asksaveasfilename(title="导出图片", defaultextension=".png",
                                                initialfile=Path(self.images[0]).name,
                                                filetypes=[("PNG", "*.png")])
            targets = [(self.images[0], dest)] if dest else []
        else:
            folder = filedialog.askdirectory(title="选择导出目录")
            targets = [(p, str(Path(folder) / Path(p).name)) for p in self.images] if folder else []
        import shutil
        for src, dst in targets:
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass


class MaxFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=IOS["bg"])
        ensure_fonts()
        self.title("最大流神经网络近似")
        self.geometry("1360x880")
        self.minsize(1080, 680)

        self.ui_queue: "queue.Queue" = queue.Queue()
        self.busy = False
        self.task_params: Dict[str, Dict[str, Any]] = {}
        self.state = dict(graph=None, dataset=None, model=None, compare=None)
        self.cards: Dict[str, Dict[str, Any]] = {}
        self.buttons: Dict[str, Any] = {}
        self.figures: List[Any] = []
        self.canvas_refs: List[Any] = []
        self.last_output: Optional[Dict[str, Any]] = None
        self.progress_t0 = time.time()

        for key, item in TASK_DEF.items():
            self.task_params[key] = {s["key"]: s["default"] for s in item["params"]}

        self.grid_rowconfigure(0, weight=1)
        # 侧边栏固定宽度（不抢额外空间），主内容区吸收全部剩余宽度
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._drain_queue)

        self._log("欢迎使用。点击左侧功能区任一按钮，在弹出面板中确认参数后即可运行。")
        self._log(f"计算设备：{str(DEVICE).upper()}    项目目录：{BASE_DIR}")
        self._refresh_cards()

    # ------------------------------------------------------------ 布局：侧边栏
    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=312, corner_radius=0, fg_color=IOS["card"],
                            border_width=1, border_color=IOS["line"])
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        side.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(side, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=20, pady=(24, 14))
        ctk.CTkLabel(head, text="最大流", font=F_TITLE, text_color=IOS["t1"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(head, text="神经网络近似 · 控制台", font=F_TINY, text_color=IOS["t3"],
                     anchor="w").pack(anchor="w", pady=(3, 0))

        hint = ctk.CTkFrame(side, fg_color=IOS["card2"], corner_radius=10)
        hint.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
        ctk.CTkLabel(hint, text="按顺序执行：生成图 → 生成数据集 → 训练 → 对比",
                     font=F_TINY, text_color=IOS["t3"], wraplength=240,
                     justify="left").pack(padx=12, pady=10)

        tasks = ctk.CTkFrame(side, fg_color="transparent")
        tasks.grid(row=2, column=0, sticky="nsew", padx=14, pady=2)
        for key in ("graph", "dataset", "train", "compare"):
            self._build_task_card(tasks, key)

        self.btn_load_model = ctk.CTkButton(
            side, text="载入已有模型 (.pth)", font=F_BTN, height=38, corner_radius=11,
            fg_color=IOS["gray"], text_color=IOS["t2"], hover_color="#E5E5EA",
            command=self.load_model_dialog)
        self.btn_load_model.grid(row=3, column=0, sticky="ew", padx=16, pady=(10, 4))

        foot = ctk.CTkFrame(side, fg_color="transparent")
        foot.grid(row=4, column=0, sticky="sew", padx=16, pady=(8, 18))
        ctk.CTkFrame(side, height=1, fg_color=IOS["line"]).grid(row=5, column=0, sticky="ew",
                                                               padx=16, pady=0)
        ctk.CTkLabel(foot, text=f"计算设备 {str(DEVICE).upper()}", font=F_TINY,
                     text_color=IOS["t3"], anchor="w").pack(anchor="w")
        ctk.CTkLabel(foot, text="模型保存目录 models/", font=F_TINY, text_color=IOS["t4"],
                     anchor="w").pack(anchor="w", pady=(3, 0))

    def _build_task_card(self, parent, key):
        item = TASK_DEF[key]
        color = item["color"]
        card = ctk.CTkFrame(parent, fg_color=IOS["card2"], corner_radius=14, border_width=1,
                            border_color=IOS["line"], cursor="hand2")
        card.pack(fill="x", pady=5)
        self.cards[key] = dict(frame=card)

        inner = ctk.CTkFrame(card, fg_color="transparent", cursor="hand2")
        inner.pack(fill="x", padx=14, pady=12)

        top = ctk.CTkFrame(inner, fg_color="transparent", cursor="hand2")
        top.pack(fill="x")
        icon = ctk.CTkFrame(top, width=34, height=34, corner_radius=17, fg_color=color,
                            cursor="hand2")
        icon.pack(side="left", padx=(0, 10))
        icon.pack_propagate(False)
        ctk.CTkLabel(icon, text=item["icon"], font=F(16, "bold"), text_color="#FFFFFF",
                     cursor="hand2").pack(expand=True)

        title_box = ctk.CTkFrame(top, fg_color="transparent", cursor="hand2")
        title_box.pack(side="left", fill="x", expand=True)
        lbl_title = ctk.CTkLabel(title_box, text=item["name"], font=F_CARD, text_color=IOS["t1"],
                                 anchor="w", cursor="hand2")
        lbl_title.pack(anchor="w")
        lbl_sub = ctk.CTkLabel(title_box, text="未运行", font=F_TINY, text_color=IOS["t4"],
                               anchor="w", cursor="hand2")
        lbl_sub.pack(anchor="w", pady=(2, 0))
        self.cards[key].update(title=lbl_title, sub=lbl_sub)

        gear = ctk.CTkLabel(top, text="›", font=F(20), text_color=IOS["t4"], cursor="hand2")
        gear.pack(side="right")

        # 参数摘要：让未运行时也能看到当前配置，卡片不再空荡
        chip = ctk.CTkFrame(inner, fg_color=IOS["gray"], corner_radius=8, cursor="hand2")
        chip.pack(fill="x", pady=(10, 0))
        lbl_chip = ctk.CTkLabel(chip, text="", font=F_TINY, text_color=IOS["t2"],
                                anchor="w", cursor="hand2", wraplength=210)
        lbl_chip.pack(fill="x", padx=9, pady=6)
        self.cards[key]["chip"] = lbl_chip

        bar = ctk.CTkProgressBar(inner, height=4, corner_radius=2, fg_color=IOS["line"],
                                 progress_color=color)
        bar.set(0)
        bar.pack(fill="x", pady=(10, 0))
        bar.pack_forget()
        self.cards[key]["bar"] = bar

        for w in (card, inner, top, title_box, lbl_title, lbl_sub, icon, gear, chip, lbl_chip):
            w.bind("<Button-1>", lambda _e, k=key: self.open_params(k))

    # ------------------------------------------------------------ 布局：主区域
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=IOS["bg"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        main.grid_rowconfigure(2, weight=1)
        main.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=26, pady=(24, 6))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="运行面板", font=F_H1, text_color=IOS["t1"],
                     anchor="w").grid(row=0, column=0, sticky="w")
        self.lbl_status = ctk.CTkLabel(head, text="空闲", font=F_SMALL, text_color=IOS["t3"],
                                       anchor="w")
        self.lbl_status.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.btn_view_output = ctk.CTkButton(
            head, text="查看输出", font=F_BTN, height=36, corner_radius=11, width=118,
            fg_color=IOS["gray"], text_color=IOS["t3"], hover_color="#E5E5EA",
            state="disabled", command=self.view_last_output)
        self.btn_view_output.grid(row=0, column=1, rowspan=2, sticky="e")

        # 总体进度卡
        prog = ctk.CTkFrame(main, fg_color=IOS["card"], corner_radius=16, border_width=1,
                            border_color=IOS["line"])
        prog.grid(row=1, column=0, sticky="ew", padx=26, pady=(6, 8))
        self.prog_frame = prog

        top = ctk.CTkFrame(prog, fg_color="transparent")
        top.pack(fill="x", padx=18, pady=(14, 8))
        self.lbl_prog_title = ctk.CTkLabel(top, text="尚未开始", font=F_CARD,
                                           text_color=IOS["t1"], anchor="w")
        self.lbl_prog_title.pack(side="left")
        self.lbl_prog_pct = ctk.CTkLabel(top, text="0%", font=F_CARD, text_color=IOS["t3"])
        self.lbl_prog_pct.pack(side="right")

        self.prog_bar = ctk.CTkProgressBar(prog, height=8, corner_radius=4,
                                           fg_color=IOS["line"], progress_color=IOS["blue"])
        self.prog_bar.set(0)
        self.prog_bar.pack(fill="x", padx=18)

        bot = ctk.CTkFrame(prog, fg_color="transparent")
        bot.pack(fill="x", padx=18, pady=(8, 14))
        self.lbl_prog_detail = ctk.CTkLabel(bot, text="等待任务", font=F_TINY,
                                            text_color=IOS["t3"], anchor="w")
        self.lbl_prog_detail.pack(side="left")
        self.lbl_prog_time = ctk.CTkLabel(bot, text="", font=F_TINY, text_color=IOS["t4"])
        self.lbl_prog_time.pack(side="right")

        body = ctk.CTkFrame(main, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=26, pady=(4, 22))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(body, fg_color=IOS["card"], corner_radius=16, border_width=1,
                            border_color=IOS["line"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        lh = ctk.CTkFrame(left, fg_color="transparent")
        lh.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(lh, text="运行日志", font=F_CARD, text_color=IOS["t1"]).pack(side="left")
        ctk.CTkButton(lh, text="清空", font=F_TINY, height=26, corner_radius=8, width=56,
                      fg_color=IOS["gray"], text_color=IOS["t2"], hover_color="#E5E5EA",
                      command=self.clear_log).pack(side="right")
        self.txt_log = ctk.CTkTextbox(left, font=F_MONO, fg_color=IOS["card2"], border_width=0,
                                      corner_radius=10, text_color=IOS["t2"])
        self.txt_log.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.txt_log.configure(state="disabled")

        right = ctk.CTkFrame(body, fg_color=IOS["card"], corner_radius=16, border_width=1,
                             border_color=IOS["line"])
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        rh = ctk.CTkFrame(right, fg_color="transparent")
        rh.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(rh, text="结果预览", font=F_CARD, text_color=IOS["t1"]).pack(side="left")
        self.lbl_preview_hint = ctk.CTkLabel(rh, text="暂无", font=F_TINY, text_color=IOS["t4"])
        self.lbl_preview_hint.pack(side="right")
        self.preview_host = ctk.CTkFrame(right, fg_color=IOS["card2"], corner_radius=12)
        self.preview_host.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._build_idle_panel(self.preview_host)

    # ------------------------------------------------------------ 空状态引导面板
    def _build_idle_panel(self, host):
        """未运行任何任务时，用 2×2 功能卡片填满主区域；每张卡片可点击直接进入对应任务。"""
        wrap = ctk.CTkScrollableFrame(host, fg_color="transparent",
                                     scrollbar_button_color=IOS["t4"],
                                     scrollbar_button_hover_color=IOS["t3"])
        wrap.pack(fill="both", expand=True, padx=6, pady=6)
        wrap.grid_columnconfigure(0, weight=1)
        wrap.grid_columnconfigure(1, weight=1)

        head = ctk.CTkFrame(wrap, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(6, 2))
        ctk.CTkLabel(head, text="工作流总览", font=F_H1, text_color=IOS["t1"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(head, text="点击任意卡片即可配置并运行该任务", font=F_TINY,
                     text_color=IOS["t3"], anchor="w").pack(anchor="w", pady=(3, 0))

        self._idle_steps = {}
        flow = [("graph", "生成网络图", "随机生成带容量有向图，并用 Edmonds-Karp 求精确最大流",
                 "确定顶点数与容量范围"),
                ("dataset", "生成数据集", "批量生成随机图并由精确算法打标签，得到训练/测试集",
                 "决定约 80% 的耗时"),
                ("train", "训练模型", "训练 MLP 拟合最大流，评估后自动保存到 models/",
                 "保存为 .pth 模型文件"),
                ("compare", "算法对比", "同一张图上对比 精确算法 与 神经网络 的结果与耗时",
                 "直观看出精度与速度")]
        for idx, (key, title, desc, tag) in enumerate(flow):
            r, c = divmod(idx, 2)
            color = TASK_DEF[key]["color"]
            card = ctk.CTkFrame(wrap, fg_color=IOS["card"], corner_radius=16,
                                border_width=1, border_color=IOS["line"], cursor="hand2")
            card.grid(row=r + 1, column=c, sticky="nsew", padx=8, pady=8)
            card.grid_propagate(False)

            # 顶部：彩色徽章 + 标题 + 箭头
            top = ctk.CTkFrame(card, fg_color="transparent", cursor="hand2")
            top.pack(fill="x", padx=16, pady=(16, 0))
            badge = ctk.CTkFrame(top, width=40, height=40, corner_radius=20,
                                 fg_color=color, cursor="hand2")
            badge.pack(side="left", padx=(0, 12))
            badge.pack_propagate(False)
            ctk.CTkLabel(badge, text=TASK_DEF[key]["icon"], font=F(18, "bold"),
                         text_color="#FFFFFF", cursor="hand2").pack(expand=True)
            tb = ctk.CTkFrame(top, fg_color="transparent", cursor="hand2")
            tb.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(tb, text=title, font=F_H2, text_color=IOS["t1"],
                         anchor="w", cursor="hand2").pack(anchor="w")
            ctk.CTkLabel(tb, text=tag, font=F_TINY, text_color=color,
                         anchor="w", cursor="hand2").pack(anchor="w", pady=(2, 0))
            arrow = ctk.CTkLabel(top, text="›", font=F(22), text_color=IOS["t4"],
                                 cursor="hand2")
            arrow.pack(side="right")

            ctk.CTkLabel(card, text=desc, font=F_SMALL, text_color=IOS["t3"],
                         anchor="w", wraplength=300, justify="left", cursor="hand2"
                         ).pack(fill="x", padx=16, pady=(12, 0))

            chip = ctk.CTkFrame(card, fg_color=IOS["card2"], corner_radius=9,
                                cursor="hand2")
            chip.pack(fill="x", padx=16, pady=(8, 0))
            lbl_chip = ctk.CTkLabel(chip, text="", font=F_TINY, text_color=IOS["t2"],
                                    anchor="w", cursor="hand2", wraplength=300)
            lbl_chip.pack(fill="x", padx=10, pady=8)
            self._idle_steps[key] = dict(card=card, color=color, chip=lbl_chip)

            for w in (card, top, tb, badge, arrow, chip, lbl_chip):
                w.bind("<Button-1>", lambda _e, k=key: self.open_params(k))

        self._idle_hint = ctk.CTkLabel(wrap, text="", font=F_TINY,
                                       text_color=IOS["t3"], anchor="w",
                                       wraplength=600, justify="left")
        self._idle_hint.grid(row=3, column=0, columnspan=2, sticky="w", padx=14,
                             pady=(6, 10))

        self._refresh_idle_panel()

    def _refresh_idle_panel(self):
        """根据已完成状态点亮卡片，并给出下一步提示。"""
        if not hasattr(self, "_idle_steps"):
            return
        try:
            if not self._idle_steps["graph"]["card"].winfo_exists():
                return
        except Exception:
            return
        order = ["graph", "dataset", "train", "compare"]
        done = {"graph": self.state["graph"] is not None,
                "dataset": self.state["dataset"] is not None,
                "train": self.state["model"] is not None,
                "compare": self.state["compare"] is not None}
        next_key = next((k for k in order if not done[k]), None)
        for key, widgets in self._idle_steps.items():
            color = widgets["color"]
            is_next = key == next_key
            card, chip = widgets["card"], widgets["chip"]
            try:
                try:
                    chip.configure(text=TASK_CHIPS[key](self.task_params[key]))
                except Exception:
                    pass
                if done[key]:
                    card.configure(fg_color=IOS["card"], border_color=color)
                    chip.configure(text="✓ " + chip.cget("text"))
                else:
                    card.configure(fg_color=IOS["card"] if is_next else IOS["card2"],
                                   border_color=color if is_next else IOS["line"])
            except Exception:
                pass
        if next_key is None:
            msg = "🎉 全部流程已完成，可在右上「查看输出」回看文字与图片结果。"
        else:
            msg = f"建议下一步：点击「{TASK_DEF[next_key]['name']}」卡片。"
        try:
            self._idle_hint.configure(text=msg)
        except Exception:
            pass

    # ------------------------------------------------------------ 日志/状态辅助
    def _log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.txt_log.configure(state="normal")
            self.txt_log.insert("end", f"[{stamp}] {text}\n")
            self.txt_log.see("end")
            self.txt_log.configure(state="disabled")
        except Exception:
            pass

    def clear_log(self):
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")

    def _set_status(self, text, color=IOS["t3"]):
        self.lbl_status.configure(text=text, text_color=color)

    def _ui(self, fn, *args, **kwargs):
        self.ui_queue.put((fn, args, kwargs))

    def _drain_queue(self):
        try:
            while True:
                fn, args, kwargs = self.ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    self._log(f"[界面更新异常] {short_err(e)}")
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

    # ------------------------------------------------------------ 任务卡片状态
    def _refresh_cards(self):
        st_map = {
            "graph": self.state["graph"],
            "dataset": self.state["dataset"],
            "train": self.state["model"],
            "compare": self.state["compare"],
        }
        subs = {
            "graph": lambda s: f"{s['nodes']} 顶点 · {s['edges']} 边 · 最大流 {s['flow']:.0f}",
            "dataset": lambda s: f"训练 {s['train']} · 测试 {s['test']} · 维度 {s['dim']}",
            "train": lambda s: (f"{Path(s['path']).name} · MAE {s['metrics']['mae']:.3f}"
                                if s.get("trained") else
                                f"{Path(s['path']).name} · 已加载"),
            "compare": lambda s: f"精确 {s['true']:.0f} vs 预测 {s['pred']:.2f}",
        }
        for key, info in self.cards.items():
            # 参数摘要 chip 始终显示当前配置
            try:
                info["chip"].configure(text=TASK_CHIPS[key](self.task_params[key]))
            except Exception:
                pass
            st = st_map[key]
            if st is not None:
                info["sub"].configure(text=subs[key](st), text_color=IOS["t2"])
                info["frame"].configure(fg_color=IOS["card"], border_color=IOS["line"])
            else:
                info["sub"].configure(text="尚未运行", text_color=IOS["t4"])
                info["frame"].configure(fg_color=IOS["card2"], border_color=IOS["line"])
        self._refresh_idle_panel()
        has_out = self._has_output()
        try:
            self.btn_view_output.configure(
                state="normal" if has_out and not self.busy else "disabled",
                text_color=IOS["blue"] if has_out and not self.busy else IOS["t3"])
        except Exception:
            pass

    def _has_output(self) -> bool:
        return self.state["model"] is not None or self.state["compare"] is not None

    def _set_card_sub(self, key, text, color=IOS["t2"]):
        self.cards[key]["sub"].configure(text=text, text_color=color)

    # ------------------------------------------------------------ 参数面板流程
    def open_params(self, key):
        if self.busy:
            self._toast("有任务正在运行，请稍候")
            return
        current = dict(self.task_params[key])

        # 参数联动：让后续的图/数据/模型维度自动保持一致，避免无谓的维度不匹配报错
        ref_graph = self.state["graph"]["params"] if self.state["graph"] else None
        ref_model = self.state["model"]
        if ref_model is not None:
            nodes = int(math.sqrt(ref_model["input_dim"]))
            if "n_nodes" in current:
                current["n_nodes"] = nodes
            if ref_graph is None:
                ref_graph = {"n_nodes": nodes}
        if ref_graph is not None:
            for k in GRAPH_KEYS:
                if k in current:
                    current[k] = ref_graph.get(k, current[k])

        # 面板通过 grab_set 实现模态；保存回调里才开始真正执行任务
        ParamDialog(self, key, current, lambda res: self._on_params_saved(key, res))

    def _on_params_saved(self, key, values):
        self.task_params[key] = values
        picked = {k: v for k, v in values.items()
                  if v is not None and not (isinstance(v, bool) and v is False)}
        pretty = ", ".join(f"{k}={v}" for k, v in list(picked.items())[:6])
        self._log(f"[{TASK_DEF[key]['name']}] 参数已保存：{pretty}")
        self.start_task(key)

    # ------------------------------------------------------------ 通用进度控制
    def _begin_task(self, key):
        self.busy = True
        self.progress_t0 = time.time()
        info = self.cards[key]
        info["bar"].pack(fill="x", pady=(10, 0))
        info["bar"].set(0)
        self._set_card_sub(key, "运行中…", IOS["blue"])
        self._set_status(f"正在执行：{TASK_DEF[key]['name']}", IOS["blue"])
        self.lbl_prog_detail.configure(text="准备中…")
        self.btn_view_output.configure(state="disabled", text_color=IOS["t3"])
        self._set_load_btn(False)

    def _set_load_btn(self, enable: bool):
        try:
            self.btn_load_model.configure(
                state="normal" if enable else "disabled",
                text_color=IOS["t2"] if enable else IOS["t4"])
        except Exception:
            pass

    def _end_task(self, key, ok=True):
        self.busy = False
        info = self.cards[key]
        try:
            info["bar"].pack_forget()
        except Exception:
            pass
        self._set_load_btn(True)
        self._refresh_cards()
        if ok:
            self._set_status("空闲", IOS["t3"])
            self.prog_bar.set(1.0)
            self.lbl_prog_title.configure(text=f"{TASK_DEF[key]['name']} · 已完成")
        else:
            self._set_status("任务失败，请查看日志", IOS["red"])
            self.lbl_prog_title.configure(text=f"{TASK_DEF[key]['name']} · 失败")

    def _set_overall(self, key, frac, detail=""):
        def do():
            frac_c = max(0.0, min(1.0, float(frac)))
            self.prog_bar.set(frac_c)
            self.cards[key]["bar"].set(frac_c)
            self.lbl_prog_pct.configure(text=f"{frac_c * 100:.0f}%")
            self.lbl_prog_title.configure(text=f"{TASK_DEF[key]['name']} · 进行中")
            if detail:
                self.lbl_prog_detail.configure(text=detail)
            elapsed = time.time() - getattr(self, "progress_t0", time.time())
            if frac_c > 0.02:
                total = elapsed / frac_c
                remain = max(0.0, total - elapsed)
                self.lbl_prog_time.configure(
                    text=f"已用 {self._fmt_dur(elapsed)} · 预计剩余 {self._fmt_dur(remain)}")
            else:
                self.lbl_prog_time.configure(text=f"已用 {self._fmt_dur(elapsed)}")
        self._ui(do)

    @staticmethod
    def _fmt_dur(sec: float) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    # ------------------------------------------------------------ 任务调度入口
    def start_task(self, key):
        runner = getattr(self, f"_task_{TASK_DEF[key]['run_key']}", None)
        if runner is None:
            self._log(f"[错误] 未实现的任务：{key}")
            return
        self._begin_task(key)
        threading.Thread(target=self._worker, args=(key, runner), daemon=True).start()

    def _worker(self, key, runner):
        try:
            runner()
        except Exception as e:
            tb = traceback.format_exc()
            self._ui(self._log, f"[{TASK_DEF[key]['name']}] 运行失败：{short_err(e)}")
            self._ui(self._log, tb)
            self._ui(self._end_task, key, False)
        finally:
            if self.busy:
                self.busy = False

    # ------------------------------------------------------------ 任务：生成图
    def _task_gen_graph(self):
        key = "graph"
        p = self.task_params["graph"]
        n = int(p["n_nodes"])
        self._set_overall(key, 0.05, "正在随机生成有向图…")
        G, cap = generate_flow_graph(n, float(p["edge_density"]),
                                     int(p["cap_min"]), int(p["cap_max"]))
        self._set_overall(key, 0.45, "正在用 Edmonds-Karp 求精确最大流…")
        flow, cost = edmonds_karp_max_flow(G)
        self._set_overall(key, 0.75, "正在绘制图像…")
        png = OUTPUT_DIR / "gui_graph.png"
        self._draw_graph(G, png)
        np.savez(GRAPH_CACHE, cap=cap)

        self.state["graph"] = dict(G=G, cap=cap, flow=float(flow),
                                   nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                                   params={k: p[k] for k in GRAPH_KEYS})
        self._set_overall(key, 1.0, "完成")
        self._ui(self._log, f"✅ 网络图生成完成：{n} 顶点 / {G.number_of_edges()} 边，"
                            f"精确最大流 = {flow:.0f}（耗时 {cost * 1000:.2f} ms）")
        self._ui(self._show_preview_graph, G, flow)
        self._ui(self._end_task, key, True)

    # ------------------------------------------------------------ 任务：数据集
    def _task_gen_dataset(self):
        key = "dataset"
        p = self.task_params["dataset"]
        self._set_overall(key, 0.02, f"准备生成 {int(p['train_samples'])} 条训练样本…")
        X_tr, y_tr = generate_dataset(int(p["train_samples"]), int(p["n_nodes"]),
                                      float(p["edge_density"]), int(p["cap_min"]),
                                      int(p["cap_max"]), edmonds_karp_max_flow)
        self._ui(self._log, f"训练集完成 {X_tr.shape}，标签均值 {float(y_tr.mean()):.2f}")
        self._set_overall(key, 0.65, f"准备生成 {int(p['test_samples'])} 条测试样本…")
        X_te, y_te = generate_dataset(int(p["test_samples"]), int(p["n_nodes"]),
                                      float(p["edge_density"]), int(p["cap_min"]),
                                      int(p["cap_max"]), edmonds_karp_max_flow)
        self._set_overall(key, 0.9, "正在保存 npz 缓存…")
        save_npz(DATASET_CACHE, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)

        png = OUTPUT_DIR / "gui_dataset_dist.png"
        self._draw_label_dist(y_tr, png)
        self.state["dataset"] = dict(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                                     train=int(len(y_tr)), test=int(len(y_te)),
                                     dim=int(X_tr.shape[1]),
                                     params={k: p.get(k) for k in GRAPH_KEYS})
        self._set_overall(key, 1.0, "完成")
        self._ui(self._log, f"✅ 数据集生成完成：训练 {len(y_tr)} / 测试 {len(y_te)}，"
                            f"特征维度 {X_tr.shape[1]}")
        self._ui(self._show_preview_dist, y_tr)
        self._ui(self._end_task, key, True)

    # ------------------------------------------------------------ 任务：训练
    def _task_train(self):
        key = "train"
        p = self.task_params["train"]
        try:
            hidden = [int(x) for x in str(p["hidden_sizes"]).split(",") if x.strip()]
        except Exception:
            raise ValueError("隐藏层结构格式错误，应为 128,64,32")
        if not hidden:
            raise ValueError("隐藏层结构不能为空")

        self._set_overall(key, 0.02, "正在准备数据集…")
        X_tr, y_tr, X_te, y_te = self._resolve_dataset(p, key)
        input_dim = int(X_tr.shape[1])
        expected = int(p["n_nodes"]) ** 2
        if expected != input_dim:
            self._ui(self._log, f"⚠️ 顶点数量参数({int(p['n_nodes'])})与数据集维度"
                                f"({input_dim})不一致，已按数据集维度 {int(math.sqrt(input_dim))} 构建模型。")

        train_loader = build_dataloader(X_tr, y_tr, int(p["batch_size"]), shuffle=True)
        test_loader = build_dataloader(X_te, y_te, int(p["batch_size"]), shuffle=False)
        model = MaxFlowMLP(input_dim=input_dim, hidden_sizes=hidden).to(DEVICE)
        params_n = sum(v.numel() for v in model.parameters())
        self._ui(self._log, f"模型结构 input={input_dim} hidden={hidden}，"
                            f"参数量 {params_n:,}，设备 {DEVICE}")
        self._set_overall(key, 0.10, "开始训练…")

        loss_curve = train_mlp_with_progress(
            model, train_loader, test_loader, float(p["lr"]), int(p["epochs"]),
            lambda frac, epoch, ep, tr, te: self._on_epoch(frac, epoch, ep, tr, te))

        self._set_overall(key, 0.88, "正在评估测试集…")
        y_pred = predict_mlp(model, X_te)
        m = calc_metrics(y_te, y_pred)

        stamp = datetime.now().strftime("%m%d_%H%M%S")
        name = str(p.get("model_name") or "maxflow_mlp").strip() or "maxflow_mlp"
        if name.endswith(".pth"):
            name = name[:-4]
        save_path = MODELS_DIR / f"{name}_{stamp}.pth"
        torch.save(model.state_dict(), save_path)
        write_csv(TRAIN_LOG_CSV, ["epoch", "train_loss", "test_loss"], loss_curve)

        png = OUTPUT_DIR / "gui_loss.png"
        self._draw_loss(loss_curve, png)

        self.state["model"] = dict(model=model, path=str(save_path), input_dim=input_dim,
                                   hidden=hidden, metrics=m, curve=loss_curve, trained=True,
                                   params={k: p.get(k) for k in ("lr", "epochs", "batch_size")})
        self._set_overall(key, 1.0, "完成")
        self._ui(self._log, f"✅ 训练完成，模型已保存到 {save_path.relative_to(BASE_DIR)}")
        self._ui(self._log, f"   测试集 MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                            f"平均相对误差={m['mean_relative_error'] * 100:.2f}%")
        self._ui(self._show_preview_loss, loss_curve)
        self._ui(self._end_task, key, True)
        self.last_output = dict(
            title=f"训练结果 · {save_path.name}",
            lines=self._train_report(p, hidden, input_dim, params_n, m, save_path, X_tr, X_te),
            images=[str(png)])

    def _resolve_dataset(self, p, key):
        use_cached = bool(p.get("use_cached", True))
        ds = self.state["dataset"]
        need_tr = int(p.get("train_samples", 0) or 0)
        need_te = int(p.get("test_samples", 0) or 0)
        gen_new = bool(p.get("gen_new", False))
        if use_cached and ds is not None and not gen_new:
            self._ui(self._log, "复用内存中的现有数据集，跳过生成步骤。")
            return ds["X_tr"], ds["y_tr"], ds["X_te"], ds["y_te"]
        if use_cached and not gen_new and DATASET_CACHE.exists():
            try:
                z = np.load(DATASET_CACHE)
                self._ui(self._log, f"复用缓存数据集 {DATASET_CACHE.name}。")
                return z["X_train"], z["y_train"], z["X_test"], z["y_test"]
            except Exception:
                pass
        self._ui(self._log, f"开始生成数据集：训练 {need_tr} / 测试 {need_te}（较慢）…")
        X_tr, y_tr = generate_dataset(need_tr, int(p["n_nodes"]), float(p["edge_density"]),
                                      int(p["cap_min"]), int(p["cap_max"]), edmonds_karp_max_flow)
        self._set_overall(key, 0.35, "训练集完成，正在生成测试集…")
        X_te, y_te = generate_dataset(need_te, int(p["n_nodes"]), float(p["edge_density"]),
                                      int(p["cap_min"]), int(p["cap_max"]), edmonds_karp_max_flow)
        save_npz(DATASET_CACHE, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
        self.state["dataset"] = dict(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                                     train=int(len(y_tr)), test=int(len(y_te)),
                                     dim=int(X_tr.shape[1]), params={})
        return X_tr, y_tr, X_te, y_te

    def _on_epoch(self, frac, epoch, total, tr, te):
        self._set_overall("train", frac, f"Epoch {epoch}/{total} · train {tr:.4f} · test {te:.4f}")

    def _train_report(self, p, hidden, input_dim, params_n, m, save_path, X_tr, X_te):
        curve = self.state["model"].get("curve") or []
        best = min(curve, key=lambda r: r[2]) if curve else None

        def g(k, default="—"):
            return p.get(k, default) if isinstance(p, dict) else default

        return [
            "=============== 训练结果报告 ===============",
            f"生成时间      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"计算设备      {DEVICE}",
            "",
            "--- 模型结构 ---",
            f"输入维度      {input_dim}  (顶点数量 {int(math.sqrt(input_dim))})",
            f"隐藏层        {hidden}",
            f"参数量        {params_n:,}",
            "",
            "--- 超参数 ---",
            f"学习率 lr     {g('lr')}",
            f"训练轮数      {g('epochs')}",
            f"批大小 batch  {g('batch_size')}",
            "",
            "--- 数据集 ---",
            f"训练样本      {len(X_tr) if X_tr is not None else '—'}",
            f"测试样本      {len(X_te) if X_te is not None else '—'}",
            "",
            "--- 测试集评估指标 ---",
            f"MAE           {m['mae']:.4f}",
            f"RMSE          {m['rmse']:.4f}",
            f"平均相对误差  {m['mean_relative_error'] * 100:.2f}%",
            "",
            "--- 训练曲线 ---",
            f"总轮数        {len(curve)}",
            f"最终 train    {curve[-1][1]:.6f}" if curve else "",
            f"最终 test     {curve[-1][2]:.6f}" if curve else "",
            f"最优 test     {best[2]:.6f}  (epoch {best[0]})" if best else "",
            "",
            "--- 模型文件 ---",
            f"保存位置      {save_path}",
            f"文件大小      {save_path.stat().st_size / 1024:.1f} KB",
            "",
            "提示：模型保存在项目根目录的 models/ 文件夹下，可在「算法对比」中直接使用。",
        ]

    # ------------------------------------------------------------ 任务：对比
    def _task_compare(self):
        key = "compare"
        if self.state["model"] is None:
            raise RuntimeError("请先完成「训练模型」或加载一个 .pth 模型")
        p = self.task_params["compare"]
        model = self.state["model"]["model"]
        input_dim = self.state["model"]["input_dim"]

        self._set_overall(key, 0.1, "正在准备对比用的图…")
        if str(p.get("source", "current")) == "current":
            if self.state["graph"] is None:
                raise RuntimeError("当前没有已生成的图，请在参数中选择「临时生成新图」")
            G = self.state["graph"]["G"]
            cap = self.state["graph"]["cap"]
            if int(cap.size) != input_dim:
                raise RuntimeError(
                    f"模型输入维度为 {input_dim}（对应 {int(math.sqrt(input_dim))} 个顶点），"
                    f"但当前图是 {int(math.sqrt(cap.size))} 个顶点。"
                    f"请重新生成一张 {int(math.sqrt(input_dim))} 顶点的图，"
                    f"或在参数中选择「临时生成新图」")
        else:
            n = int(p["n_nodes"])
            if n * n != input_dim:
                raise RuntimeError(
                    f"模型输入维度为 {input_dim}（对应 {int(math.sqrt(input_dim))} 个顶点），"
                    f"与参数中的顶点数量 {n} 不一致")
            G, cap = generate_flow_graph(n, float(p["edge_density"]),
                                         int(p["cap_min"]), int(p["cap_max"]))

        self._set_overall(key, 0.4, "正在执行精确算法与神经网络推理…")
        res = compare_single_graph(G, cap, model)
        save_json(res, OUTPUT_DIR / "gui_single_compare.json")
        self._set_overall(key, 0.75, "正在绘制对比图…")
        png = OUTPUT_DIR / "gui_compare.png"
        self._draw_compare(res, png)

        self.state["compare"] = dict(res=res, **{k: res[k] for k in
                                                 ("true_max_flow", "pred_max_flow")})
        self.state["compare"]["true"] = res["true_max_flow"]
        self.state["compare"]["pred"] = res["pred_max_flow"]
        self._set_overall(key, 1.0, "完成")
        if res["time_nn_sec"] > 0:
            speed = res["time_exact_sec"] / res["time_nn_sec"]
            tail = f"，神经网络约为精确算法的 {speed:.1f}× 速度"
        else:
            tail = ""
        self._ui(self._log, f"✅ 对比完成：精确 {res['true_max_flow']} vs 预测 "
                            f"{res['pred_max_flow']:.2f}，绝对误差 {res['abs_error']:.4f}"
                            f"{tail}")
        self._ui(self._show_preview_compare, res)
        self._ui(self._end_task, key, True)
        self.last_output = dict(
            title="算法对比结果",
            lines=self._compare_report(res, G),
            images=[str(png)])

    def _compare_report(self, res, G):
        return [
            "============= 算法对比报告 =============",
            f"生成时间      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"图规模        {G.number_of_nodes()} 顶点 / {G.number_of_edges()} 边",
            "",
            "--- 结果 ---",
            f"精确最大流    {res['true_max_flow']}",
            f"神经网络预测  {res['pred_max_flow']}",
            f"绝对误差      {res['abs_error']}",
            f"相对误差      {res['rel_error'] * 100:.2f}%",
            "",
            "--- 耗时 ---",
            f"Edmonds-Karp  {res['time_exact_sec'] * 1e3:.4f} ms",
            f"神经网络推理  {res['time_nn_sec'] * 1e3:.4f} ms",
            f"加速比        {res['time_exact_sec'] / res['time_nn_sec']:.2f}×"
            if res["time_nn_sec"] else "",
            "",
            "结论：神经网络单次推理远快于精确算法，适合需要大量重复估计最大流的场景；",
            "      代价是存在一定近似误差，可通过增加训练样本与轮数进一步降低。",
        ]

    # ------------------------------------------------------------ 查看输出
    def view_last_output(self):
        out = self.last_output
        if out is None:
            self._toast("暂无可查看的输出，请先完成训练或对比")
            return
        OutputDialog(self, out["title"], [x for x in out["lines"] if x != ""],
                     out.get("images", []))

    def _toast(self, text):
        try:
            win = ctk.CTkToplevel(self)
            win.overrideredirect(True)
            win.configure(fg_color=IOS["card"])
            win.attributes("-topmost", True)
            ctk.CTkLabel(win, text=text, font=F_SMALL, text_color=IOS["t1"]).pack(padx=20, pady=12)
            win.update_idletasks()
            w, h = win.winfo_reqwidth(), win.winfo_reqheight()
            x = self.winfo_x() + (self.winfo_width() - w) // 2
            y = self.winfo_y() + 70
            win.geometry(f"{w}x{h}+{x}+{y}")
            win.after(1800, win.destroy)
        except Exception:
            pass

    # ------------------------------------------------------------ 绘图（工作线程）
    def _new_fig(self, w=7.2, h=4.8):
        fig = Figure(figsize=(w, h), dpi=115)
        fig.patch.set_facecolor(CHART["paper"])
        return fig

    # ------------------------------------------------------------ 图表美化工件
    @staticmethod
    def _style_ax(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
        """统一的 iOS 风坐标区：去边框、细网格、柔和字色。"""
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(CHART["grid"])
            ax.spines[sp].set_linewidth(1.1)
        ax.set_facecolor(CHART["paper"])
        ax.tick_params(colors=CHART["ink3"], labelsize=9, length=0, pad=6)
        if grid_axis in ("y", "both"):
            ax.grid(axis="y", alpha=0.55, linestyle="-", linewidth=0.8,
                    color=CHART["grid"])
        if grid_axis in ("x", "both"):
            ax.grid(axis="x", alpha=0.45, linestyle="-", linewidth=0.8,
                    color=CHART["grid"])
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, fontsize=13.5, color=CHART["ink"], pad=14,
                         fontweight="bold", loc="left")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=CHART["ink2"], labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=CHART["ink2"], labelpad=8)
        return ax

    @staticmethod
    def _chip(ax, x, y, text, color, bg=None, fontsize=8.5, ha="left"):
        """图上的小标签胶囊（iOS 徽章感）。"""
        ax.annotate(
            text, xy=(x, y), xycoords="axes fraction", ha=ha, va="center",
            fontsize=fontsize, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42", facecolor=bg or (color + "1A"),
                      edgecolor="none"),
        )

    def _savefig(self, fig, png: Path, pad=0.24):
        # 含手动放置的 colorbar 轴时 tight_layout 会失效，改成受控的子图间距
        try:
            fig.tight_layout(pad=pad)
        except Exception:
            pass
        fig.savefig(png, dpi=145, facecolor=CHART["paper"], bbox_inches="tight")
        plt.close(fig)

    def _draw_graph(self, G, png: Path):
        """源点/汇点分层的网络图：按容量着色的渐变、粗细随容量、高光源汇点。"""
        fig = self._new_fig(8.0, 5.6)
        # 预留右侧一列给容量色条，避免和 tight_layout 冲突
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.028], wspace=0.04,
                              left=0.05, right=0.93, top=0.88, bottom=0.10)
        ax = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[0, 1])
        ax.set_facecolor(CHART["panel"])
        t_node = max(G.nodes)

        # 按「层」排布：源点在最左、汇点在最右，比弹簧布局更有结构感
        try:
            dist_from_s = nx.single_source_shortest_path_length(G, 0)
        except Exception:
            dist_from_s = {}
        layers: Dict[int, List[int]] = {}
        for nd in G.nodes:
            if nd == t_node:
                lv = 99
            else:
                lv = dist_from_s.get(nd, 1)
            layers.setdefault(lv, []).append(nd)
        ordered = sorted(layers.keys())
        pos = {}
        for xi, lv in enumerate(ordered):
            members = layers[lv]
            x = xi / max(1, len(ordered) - 1)
            for yi, nd in enumerate(members):
                pos[nd] = (x, (yi - (len(members) - 1) / 2) / max(1, len(members)) * 0.8)
        # 补上孤立点，避免 KeyError
        for nd in G.nodes:
            pos.setdefault(nd, (0.5, 0.0))

        caps = [d["capacity"] for _, _, d in G.edges(data=True)]
        cmin, cmax = (min(caps), max(caps)) if caps else (1, 1)
        norm = matplotlib.colors.Normalize(vmin=cmin, vmax=max(cmax, cmin + 1))

        # 边的颜色/粗细都随容量变化
        edge_colors, edge_widths = [], []
        for _, _, d in G.edges(data=True):
            edge_colors.append(CMAP_MAIN(norm(d["capacity"])))
            edge_widths.append(0.9 + 2.1 * norm(d["capacity"]))

        nx.draw_networkx_edges(
            G, pos, arrowstyle="-|>", arrowsize=15, edge_color=edge_colors,
            width=edge_widths, alpha=0.88, node_size=760, ax=ax,
            connectionstyle="arc3,rad=0.07",
            node_shape="o", min_source_margin=14, min_target_margin=14)

        nc = [CHART["green"] if nd == 0 else CHART["pink"] if nd == t_node
              else CHART["blue"] for nd in G.nodes]
        sizes = [980 if nd in (0, t_node) else 620 for nd in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=sizes, ax=ax,
                               edgecolors="#FFFFFF", linewidths=2.4)
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=10,
                                font_weight="bold", ax=ax)

        labels = {(u, v): d["capacity"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=labels, font_size=7, font_color=CHART["ink2"], ax=ax,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#FFFFFF",
                      edgecolor="none", alpha=0.85))

        ax.set_title("流量网络拓扑", fontsize=14, color=CHART["ink"], pad=16,
                     fontweight="bold", loc="left")
        ax.set_xlabel(f"源点 0 → 汇点 {t_node}   ·   {G.number_of_nodes()} 顶点 / "
                      f"{G.number_of_edges()} 条边", fontsize=10, color=CHART["ink2"],
                      labelpad=10)
        ax.axis("off")
        ax.margins(0.12)

        # 图例 + 容量色条
        handles = [
            matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=10,
                                    markerfacecolor=CHART["green"], markeredgecolor="#FFFFFF",
                                    markeredgewidth=1.6, label="源点 s"),
            matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=10,
                                    markerfacecolor=CHART["blue"], markeredgecolor="#FFFFFF",
                                    markeredgewidth=1.6, label="中间节点"),
            matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=12,
                                    markerfacecolor=CHART["pink"], markeredgecolor="#FFFFFF",
                                    markeredgewidth=1.6, label="汇点 t"),
        ]
        leg = ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9,
                        facecolor="#FFFFFF", edgecolor=CHART["grid"], borderpad=0.7)
        for t in leg.get_texts():
            t.set_color(CHART["ink2"])

        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP_MAIN),
            cax=cbar_ax, orientation="vertical")
        cbar.set_label("边容量", fontsize=9, color=CHART["ink2"], labelpad=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8, colors=CHART["ink3"], length=0)

        fig.savefig(png, dpi=145, facecolor=CHART["paper"], bbox_inches="tight")
        plt.close(fig)

    def _kde_curve(self, y):
        """高斯核密度估计，得到平滑的分布曲线(不需要额外依赖 scipy 之外的东西)。"""
        try:
            from scipy.stats import gaussian_kde
            if len(y) < 3 or float(np.std(y)) < 1e-9:
                return None, None
            kde = gaussian_kde(np.asarray(y, dtype=float))
            lo, hi = float(np.min(y)), float(np.max(y))
            pad = max((hi - lo) * 0.08, 0.5)
            xs = np.linspace(lo - pad, hi + pad, 220)
            return xs, kde(xs)
        except Exception:
            return None, None

    def _draw_label_dist(self, y, png: Path):
        """双面板：左=带 KDE 的直方图 + 分位线，右=箱线图与离群点。"""
        y = np.asarray(y, dtype=float)
        fig = self._new_fig(8.6, 4.0)
        gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 1, 0.02], wspace=0.28,
                              left=0.07, right=0.97, top=0.86, bottom=0.15)

        # ---- 左：直方图 + KDE
        ax = fig.add_subplot(gs[0, 0])
        self._style_ax(ax, title="最大流标签分布", xlabel="最大流数值", ylabel="样本数")
        counts, bin_edges = np.histogram(y, bins=min(26, max(6, len(np.unique(y)))))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = np.diff(bin_edges) * 0.86
        ax.bar(centers, counts, width=widths, color=CHART["blue"], alpha=0.82,
               edgecolor="none", zorder=2, label="频数")

        xs, dens = self._kde_curve(y)
        if xs is not None:
            # KDE 密度换算到「样本数」尺度，才能和直方图同轴
            scale = len(y) * np.diff(bin_edges)[0]
            ax.plot(xs, dens * scale, color=CHART["purple"], linewidth=2.3, zorder=4,
                    label="核密度估计")
            ax.fill_between(xs, dens * scale, color=CHART["purple"], alpha=0.13, zorder=1)

        mean = float(np.mean(y))
        med = float(np.median(y))
        ax.axvline(mean, color=CHART["pink"], linestyle="--", linewidth=1.6, zorder=5,
                   label=f"均值 {mean:.1f}")
        ax.axvline(med, color=CHART["green"], linestyle=":", linewidth=1.6, zorder=5,
                   label=f"中位数 {med:.1f}")
        leg = ax.legend(frameon=True, fontsize=8.5, facecolor="#FFFFFF",
                        edgecolor=CHART["grid"], loc="upper right")
        for t in leg.get_texts():
            t.set_color(CHART["ink2"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # ---- 右：箱线图 + 抖动散点
        ax2 = fig.add_subplot(gs[0, 1])
        for sp in ("top", "right", "bottom"):
            ax2.spines[sp].set_visible(False)
        ax2.spines["left"].set_color(CHART["grid"])
        ax2.set_facecolor(CHART["paper"])
        ax2.tick_params(colors=CHART["ink3"], labelsize=9, length=0)
        ax2.set_xticks([])
        bp = ax2.boxplot(y, vert=True, widths=0.5, patch_artist=True,
                         medianprops=dict(color=CHART["pink"], linewidth=2.2),
                         boxprops=dict(facecolor=CHART["cyan"], alpha=0.35,
                                       edgecolor=CHART["cyan"], linewidth=1.6),
                         whiskerprops=dict(color=CHART["ink3"], linewidth=1.3),
                         capprops=dict(color=CHART["ink3"], linewidth=1.3),
                         flierprops=dict(marker="o", markersize=3.5,
                                         markerfacecolor=CHART["orange"],
                                         markeredgecolor="none", alpha=0.7))
        rng = np.random.default_rng(7)
        jitter = rng.normal(1.0, 0.055, size=len(y))
        ax2.scatter(jitter, y, s=7, color=CHART["blue"], alpha=0.28, zorder=3,
                    edgecolors="none")
        ax2.set_ylabel("最大流", fontsize=10, color=CHART["ink2"], labelpad=8)
        ax2.set_title("离群点检查", fontsize=11.5, color=CHART["ink"], pad=12,
                      fontweight="bold", loc="left")

        # 顶部统计胶囊
        std = float(np.std(y))
        fig.text(0.075, 0.955, f"样本 {len(y)}", fontsize=9, color=CHART["ink2"],
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=CHART["blue"] + "18",
                           edgecolor="none"))
        fig.text(0.175, 0.955, f"均值 {mean:.1f} ± {std:.1f}", fontsize=9,
                 color=CHART["ink2"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=CHART["purple"] + "18",
                           edgecolor="none"))
        fig.text(0.30, 0.955, f"范围 {y.min():.0f} ~ {y.max():.0f}", fontsize=9,
                 color=CHART["ink2"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=CHART["teal"] + "18",
                           edgecolor="none"))

        fig.suptitle("数据集概览", fontsize=14, color=CHART["ink"], fontweight="bold",
                     x=0.068, ha="left", y=0.995)
        fig.savefig(png, dpi=145, facecolor=CHART["paper"], bbox_inches="tight")
        plt.close(fig)

    def _draw_loss(self, curve, png: Path):
        """双面板：左=对数损失曲线 + 填充差值带，右=相对差距/Gap 与最佳点标注。"""
        fig = self._new_fig(9.0, 4.3)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.26,
                              left=0.065, right=0.975, top=0.83, bottom=0.14)
        ax = fig.add_subplot(gs[0, 0])

        if not curve:
            self._style_ax(ax, title="训练损失曲线")
            ax.text(0.5, 0.5, "暂无训练记录", ha="center", va="center",
                    fontsize=12, color=CHART["ink3"], transform=ax.transAxes)
            fig.savefig(png, dpi=145, facecolor=CHART["paper"], bbox_inches="tight")
            plt.close(fig)
            return

        ep = [int(r[0]) for r in curve]
        tr = np.array([float(r[1]) for r in curve])
        te = np.array([float(r[2]) for r in curve])
        best_i = int(np.argmin(te))
        pos_mask = tr > 0

        # ---- 左：损失曲线（对数纵轴，收敛细节更清楚）
        use_log = bool(pos_mask.all()) and (tr.min() > 0) and (tr.max() / max(tr.min(), 1e-12) > 20)
        self._style_ax(
            ax, title="训练 / 测试损失收敛曲线", xlabel="Epoch",
            ylabel="MSE Loss（对数轴）" if use_log else "MSE Loss", grid_axis="both")

        ax.plot(ep, tr, color=CHART["blue"], linewidth=2.4, label="Train loss",
                solid_capstyle="round", zorder=3)
        ax.plot(ep, te, color=CHART["orange"], linewidth=2.4, label="Test loss",
                solid_capstyle="round", zorder=3)
        # 填充 train→test 的差值带＝泛化间隙，视觉上一眼看出过拟合程度
        ax.fill_between(ep, tr, te, color=CHART["purple"], alpha=0.13, zorder=1,
                        interpolate=True)

        if use_log:
            ax.set_yscale("log")

        # 最佳点 + 末点标注
        ax.scatter([ep[best_i]], [te[best_i]], s=90, facecolor="#FFFFFF",
                   edgecolor=CHART["pink"], linewidth=2.4, zorder=6)
        ax.annotate(f"最优 test\n{te[best_i]:.4f}",
                    xy=(ep[best_i], te[best_i]),
                    xytext=(0, 18), textcoords="offset points", ha="center",
                    fontsize=8.5, color=CHART["pink"], fontweight="bold")
        ax.scatter([ep[-1]], [te[-1]], s=52, color=CHART["orange"], zorder=6,
                   edgecolors="#FFFFFF", linewidths=1.6)

        leg = ax.legend(frameon=True, fontsize=9, facecolor="#FFFFFF",
                        edgecolor=CHART["grid"], loc="upper right")
        for t in leg.get_texts():
            t.set_color(CHART["ink2"])

        # ---- 右：泛化间隙（test-train，对数）
        ax2 = fig.add_subplot(gs[0, 1])
        gap = te - tr
        self._style_ax(ax2, title="泛化间隙", xlabel="Epoch",
                       ylabel="test − train", grid_axis="both")
        ax2.axhline(0, color=CHART["ink3"], linewidth=1.1, alpha=0.7)
        ax2.plot(ep, gap, color=CHART["purple"], linewidth=2.2, solid_capstyle="round")
        ax2.fill_between(ep, gap, 0, where=gap >= 0, color=CHART["pink"], alpha=0.16,
                         interpolate=True)
        ax2.fill_between(ep, gap, 0, where=gap < 0, color=CHART["green"], alpha=0.16,
                         interpolate=True)

        improve = ((tr[0] - tr[-1]) / tr[0] * 100) if tr[0] else 0.0
        fig.text(0.068, 0.945, f"{len(ep)} epochs", fontsize=9, color=CHART["ink2"],
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=CHART["blue"] + "18",
                           edgecolor="none"))
        fig.text(0.163, 0.945, f"损失下降 {improve:.1f}%", fontsize=9, color=CHART["ink2"],
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=CHART["green"] + "18",
                           edgecolor="none"))
        gap_end = float(gap[-1])
        verdict = "存在过拟合" if gap_end > 0 else "拟合良好"
        vc = CHART["pink"] if gap_end > 0 else CHART["green"]
        fig.text(0.288, 0.945, f"{verdict} ({gap_end:+.4f})", fontsize=9, color=CHART["ink2"],
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=vc + "20", edgecolor="none"))

        fig.suptitle("训练过程分析", fontsize=14.5, color=CHART["ink"], fontweight="bold",
                     x=0.065, ha="left", y=0.99)
        fig.savefig(png, dpi=145, facecolor=CHART["paper"], bbox_inches="tight")
        plt.close(fig)

    def _draw_compare(self, res, png: Path):
        """三面板：精度半环仪表 + 结果对比条 + 耗时对数轴。"""
        true_v = float(res["true_max_flow"])
        pred_v = float(res["pred_max_flow"])
        rel = float(res.get("rel_error", 0.0))
        accuracy = max(0.0, 1.0 - rel)
        t_exact = float(res["time_exact_sec"])
        t_nn = float(res["time_nn_sec"])

        fig = self._new_fig(9.2, 4.1)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.15], wspace=0.34,
                              left=0.055, right=0.975, top=0.82, bottom=0.16)

        # ---- 左：半环仪表（预测精度）
        ax = fig.add_subplot(gs[0, 0], projection="polar")
        ax.set_facecolor(CHART["paper"])
        n_seg = 240
        theta_full = np.linspace(np.pi, 0, n_seg)
        ax.plot(theta_full, np.ones(n_seg), color=CHART["grid"], linewidth=13,
                solid_capstyle="round")
        acc_color = CMAP_MAIN(0.25 + 0.75 * accuracy)
        n_fill = max(1, int(n_seg * min(1.0, accuracy)))
        ax.plot(theta_full[:n_fill], np.ones(n_fill), color=acc_color, linewidth=13,
                solid_capstyle="round", zorder=3)
        ax.plot([np.pi - np.pi * accuracy], [1.0], marker="o", markersize=9,
                color="#FFFFFF", markeredgecolor=acc_color, markeredgewidth=2.6, zorder=5)
        ax.set_ylim(0, 1.35)
        ax.set_xlim(0, np.pi)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(0)
        ax.set_axis_off()
        ax.text(np.pi / 2, 0.62, f"{accuracy * 100:.1f}%", ha="center", va="center",
                fontsize=25, fontweight="bold", color=CHART["ink"])
        ax.text(np.pi / 2, 0.30, "预测精度", ha="center", va="center", fontsize=10.5,
                color=CHART["ink2"])
        ax.text(np.pi / 2, 0.10, f"相对误差 {rel * 100:.2f}%", ha="center", va="center",
                fontsize=8.5, color=CHART["ink3"])
        ax.set_title("神经网络 vs 精确解", fontsize=12.5, color=CHART["ink"], pad=6,
                     fontweight="bold")

        # ---- 中：结果对比条
        ax2 = fig.add_subplot(gs[0, 1])
        self._style_ax(ax2, title="最大流结果对比", ylabel="最大流数值")
        names = ["精确解", "神经网络"]
        vals = [true_v, pred_v]
        bars = ax2.bar(names, vals, width=0.5, color=[CHART["blue"], CHART["indigo"]],
                       edgecolor="none", zorder=3)
        for b in bars:
            b.set_linewidth(0)
        for b, v, c in zip(bars, vals, [CHART["blue"], CHART["indigo"]]):
            ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                     va="bottom", fontsize=11.5, fontweight="bold", color=c)
        top_v = max(vals + [1e-9])
        ax2.set_ylim(0, top_v * 1.22)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, fontsize=10, color=CHART["ink2"])
        if true_v > 0:
            ax2.hlines(pred_v, -0.42, 0.42, colors=CHART["pink"], linestyles="--",
                       linewidth=1.4, zorder=4)
            ax2.text(0.5, top_v * 1.06, f"绝对误差 {float(res['abs_error']):.3f}",
                     ha="center", fontsize=9, color=CHART["pink"], fontweight="bold")

        # ---- 右：耗时对比（对数轴，差距往往是数量级的）
        ax3 = fig.add_subplot(gs[0, 2])
        self._style_ax(ax3, title="单次求解耗时", ylabel="耗时 (ms，对数轴)",
                       grid_axis="y")
        ms = [max(t_exact * 1e3, 1e-6), max(t_nn * 1e3, 1e-6)]
        names3 = ["Edmonds-Karp", "神经网络推理"]
        bars3 = ax3.barh(names3, ms, height=0.46,
                         color=[CHART["teal"], CHART["orange"]], edgecolor="none",
                         zorder=3)
        ax3.set_xscale("log")
        for b, v in zip(bars3, ms):
            ax3.text(v * 1.25, b.get_y() + b.get_height() / 2,
                     f"{v * 1e3:.3f} µs" if v < 1 else f"{v:.4f} ms",
                     va="center", fontsize=8.5, color=CHART["ink2"], fontweight="bold")
        ax3.tick_params(axis="y", labelsize=9.5)
        ax3.set_xlim(max(min(ms) / 8, 1e-7), max(ms) * 12)
        for lbl in ax3.get_yticklabels():
            lbl.set_color(CHART["ink2"])

        if t_nn > 0:
            speed = t_exact / t_nn
            fig.text(0.055, 0.945, f"加速 {speed:.1f}×", fontsize=10,
                     color=CHART["ink"], fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.42", facecolor=CHART["green"] + "22",
                               edgecolor="none"))
        fig.suptitle("精确算法 vs 神经网络", fontsize=14.5, color=CHART["ink"],
                     fontweight="bold", x=0.055, ha="left", y=0.99)
        fig.savefig(png, dpi=145, facecolor=CHART["paper"], bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------ 预览（主线程）
    def _clear_preview(self):
        for w in self.preview_host.winfo_children():
            w.destroy()
        self.canvas_refs.clear()

    def _embed_fig(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.preview_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas_refs.append(canvas)
        return canvas

    def _show_preview_graph(self, G, flow):
        self._clear_preview()
        fig = self._new_fig(6.4, 4.6)
        ax = fig.add_subplot(111)
        t_node = max(G.nodes)
        pos = nx.spring_layout(G, seed=42)
        colors = [IOS["green"] if nd == 0 else IOS["red"] if nd == t_node else IOS["blue"]
                  for nd in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=420, ax=ax,
                               edgecolors="#FFFFFF", linewidths=1.6)
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=9,
                                font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=12, edge_color="#B6BBC4",
                               width=1.2, alpha=0.9, node_size=420,
                               connectionstyle="arc3,rad=0.06", ax=ax)
        ax.set_title(f"最大流 = {flow:.0f}", fontsize=12, color=IOS["t2"], pad=10)
        ax.axis("off")
        fig.tight_layout()
        self._embed_fig(fig)
        self.lbl_preview_hint.configure(text="点击查看完整输出")

    def _show_preview_dist(self, y_tr):
        self._clear_preview()
        fig = self._new_fig(6.4, 4.6)
        ax = fig.add_subplot(111)
        counts = np.bincount(np.round(y_tr).astype(int))
        ax.bar(np.arange(len(counts)), counts, color=IOS["green"], width=0.72)
        if len(y_tr):
            mean = float(np.mean(y_tr))
            ax.axvline(mean, color=IOS["red"], linestyle="--", linewidth=1.4,
                       label=f"均值 {mean:.1f}")
            ax.legend(frameon=False, fontsize=9, labelcolor=IOS["t2"])
        ax.set_title("训练集标签分布", fontsize=12, color=IOS["t2"], pad=10)
        ax.set_xlabel("最大流", fontsize=10, color=IOS["t3"])
        ax.set_ylabel("样本数", fontsize=10, color=IOS["t3"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25, color=IOS["line"])
        fig.tight_layout()
        self._embed_fig(fig)
        self.lbl_preview_hint.configure(text="训练集分布")

    def _show_preview_loss(self, curve):
        self._clear_preview()
        fig = self._new_fig(6.4, 4.6)
        ax = fig.add_subplot(111)
        if curve:
            ep = [r[0] for r in curve]
            ax.plot(ep, [r[1] for r in curve], label="Train", color=IOS["blue"])
            ax.plot(ep, [r[2] for r in curve], label="Test", color=IOS["orange"])
            ax.legend(frameon=False, fontsize=9, labelcolor=IOS["t2"])
        ax.set_title("训练损失曲线", fontsize=12, color=IOS["t2"], pad=10)
        ax.set_xlabel("Epoch", fontsize=10, color=IOS["t3"])
        ax.set_ylabel("MSE", fontsize=10, color=IOS["t3"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25, color=IOS["line"])
        fig.tight_layout()
        self._embed_fig(fig)
        self.lbl_preview_hint.configure(text="点击查看完整输出")

    def _show_preview_compare(self, res):
        self._clear_preview()
        fig = self._new_fig(6.4, 4.6)
        ax = fig.add_subplot(111)
        names = ["精确", "神经网络"]
        vals = [res["true_max_flow"], res["pred_max_flow"]]
        bars = ax.bar(names, vals, color=[IOS["blue"], IOS["indigo"]], width=0.44)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f" {v:.2f}", ha="center", va="bottom",
                    fontsize=10, color=IOS["t2"])
        ax.set_title(f"误差 {res['abs_error']:.4f} · 加速 "
                     f"{res['time_exact_sec'] / res['time_nn_sec']:.1f}×"
                     if res["time_nn_sec"] else "对比",
                     fontsize=11, color=IOS["t2"], pad=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25, color=IOS["line"])
        fig.tight_layout()
        self._embed_fig(fig)
        self.lbl_preview_hint.configure(text="点击查看完整输出")

    def load_model_dialog(self):
        path = filedialog.askopenfilename(title="选择模型文件", filetypes=[("PyTorch", "*.pth")])
        if not path:
            return
        self._load_model_file(path)

    @staticmethod
    def _infer_arch(state: dict):
        """从 state_dict 推断 (input_dim, hidden_sizes)。

        MaxFlowMLP 的结构是 Linear/ReLU 交替，所以 net.{偶数}.weight 全是二维权重；
        除最后一层外，每一层的输出维度就是下一层的隐藏单元数。
        """
        weights = []
        for key in sorted(state.keys(), key=lambda k: int(k.split(".")[1])
                          if k.split(".")[1].isdigit() else -1):
            if "weight" in key and hasattr(state[key], "dim") and state[key].dim() == 2:
                weights.append(state[key])
        if not weights:
            raise RuntimeError("无法从权重推断网络结构")
        input_dim = int(weights[0].shape[1])
        hidden = [int(w.shape[0]) for w in weights[:-1]]
        return input_dim, hidden

    def _load_model_file(self, path):
        try:
            state = torch.load(path, map_location=DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            input_dim, hidden = self._infer_arch(state)
            model = MaxFlowMLP(input_dim=input_dim, hidden_sizes=hidden).to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            self.state["model"] = dict(model=model, path=path, input_dim=input_dim,
                                       hidden=hidden,
                                       metrics=dict(mae=0.0, rmse=0.0, mean_relative_error=0.0),
                                       curve=[], params={}, trained=False)
            nodes_n = int(math.sqrt(input_dim))
            self._log(f"✅ 已加载模型 {Path(path).name}（自动推断结构：输入维度 "
                      f"{input_dim} / 隐藏层 {hidden}，对应 {nodes_n} 个顶点）")
            self.state["compare"] = None
            self._refresh_cards()

            # 同步模型维度到下游任务参数，避免对比时维度不匹配
            for key in ("compare", "dataset", "train"):
                if "n_nodes" in self.task_params[key]:
                    self.task_params[key]["n_nodes"] = nodes_n
            self._refresh_cards()

            need_graph = self.state["graph"] is None or \
                self.state["graph"]["cap"].size != input_dim
            self._offer_next_step(path, nodes_n, need_graph)
        except Exception as e:
            self._log(f"❌ 加载模型失败：{short_err(e)}")
            self._log("提示：确认选择的是本项目 MaxFlowMLP 训练出的 .pth 文件。")

    def _offer_next_step(self, path, nodes_n, need_graph):
        """载入模型后弹出「下一步」引导，避免加载完没有后续动作。"""
        top = ctk.CTkToplevel(self, fg_color=IOS["bg"])
        top.title("")
        top.configure(fg_color=IOS["bg"])
        top.transient(self)
        top.resizable(False, False)
        try:
            top.attributes("-topmost", True)
        except Exception:
            pass

        head = ctk.CTkFrame(top, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(22, 8))
        badge = ctk.CTkFrame(head, width=44, height=44, corner_radius=22,
                             fg_color=IOS["indigo"])
        badge.pack(side="left", padx=(0, 12))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text="✓", font=F(22, "bold"), text_color="#FFFFFF").pack(expand=True)
        box = ctk.CTkFrame(head, fg_color="transparent")
        box.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(box, text="模型已就绪", font=F_H2, text_color=IOS["t1"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(box, text=Path(path).name, font=F_TINY, text_color=IOS["t3"],
                     anchor="w").pack(anchor="w", pady=(2, 0))

        card = ctk.CTkFrame(top, fg_color=IOS["card"], corner_radius=14, border_width=1,
                            border_color=IOS["line"])
        card.pack(fill="x", padx=22, pady=(4, 6))
        ctk.CTkLabel(card, text="下一步可以做什么", font=F_CARD, text_color=IOS["t1"],
                     anchor="w").pack(anchor="w", padx=16, pady=(14, 8))

        if need_graph:
            steps = [("①", f"先生成一张 {nodes_n} 顶点的图",
                      "模型输入维度要求图的顶点数一致", IOS["teal"], "graph"),
                     ("②", "再运行「算法对比」",
                      "对比精确算法与神经网络的精度和耗时", IOS["indigo"], "compare")]
        else:
            steps = [("①", "运行「算法对比」",
                      f"当前图正好是 {nodes_n} 顶点，可直接对比", IOS["indigo"], "compare"),
                     ("②", "重新训练并保存新模型",
                      "用自己的数据集微调出更好的效果", IOS["blue"], "train")]

        for tag, title, desc, color, act in steps:
            row = ctk.CTkFrame(card, fg_color="transparent", cursor="hand2")
            row.pack(fill="x", padx=14, pady=5)
            dot = ctk.CTkFrame(row, width=28, height=28, corner_radius=14,
                               fg_color=color, cursor="hand2")
            dot.pack(side="left", padx=(0, 10))
            dot.pack_propagate(False)
            ctk.CTkLabel(dot, text=tag, font=F_TINY, text_color="#FFFFFF",
                         cursor="hand2").pack(expand=True)
            tb = ctk.CTkFrame(row, fg_color="transparent", cursor="hand2")
            tb.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(tb, text=title, font=F_BODY, text_color=IOS["t1"], anchor="w",
                         cursor="hand2").pack(anchor="w")
            ctk.CTkLabel(tb, text=desc, font=F_TINY, text_color=IOS["t3"], anchor="w",
                         cursor="hand2", wraplength=300).pack(anchor="w", pady=(1, 0))
            for w in (row, dot, tb) + tuple(tb.winfo_children()):
                w.bind("<Button-1>",
                       lambda _e, a=act: (top.destroy(), self.after(60, self.open_params, a)))

        foot = ctk.CTkFrame(top, fg_color="transparent")
        foot.pack(fill="x", padx=22, pady=(6, 20))
        ctk.CTkButton(foot, text="稍后自己操作", font=F_BTN, height=40, corner_radius=12,
                      fg_color=IOS["gray"], text_color=IOS["t2"], hover_color="#E5E5EA",
                      command=top.destroy).pack(fill="x")

        top.update_idletasks()
        w, h = 480, 300 if need_graph else 300
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        mx, my = self.winfo_x(), self.winfo_y()
        mw, mh = self.winfo_width(), self.winfo_height()
        if mw > 10 and mh > 10:
            x, y = mx + (mw - w) // 2, my + (mh - h) // 2
        else:
            x, y = (sw - w) // 2, (sh - h) // 2
        top.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")
        top.lift()
        top.focus_force()

    def _on_close(self):
        try:
            self.quit()
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------- 带进度的训练
def train_mlp_with_progress(model, train_loader, test_loader, lr, epochs, on_epoch):
    """与 nn_approximator.train_mlp 等价，但额外回调 epoch 级进度。"""
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    log = []
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        tr = total / max(1, len(train_loader))

        model.eval()
        te = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                te += criterion(model(xb), yb).item()
        te /= max(1, len(test_loader))
        model.train()

        log.append([ep + 1, tr, te])
        frac = 0.10 + 0.78 * (ep + 1) / max(1, epochs)
        try:
            on_epoch(frac, ep + 1, epochs, tr, te)
        except Exception:
            pass
    return log


if __name__ == "__main__":
    app = MaxFlowApp()
    app.mainloop()
