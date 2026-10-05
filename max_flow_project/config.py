"""
项目全局配置
"""
import torch
from pathlib import Path

# ========= 路径配置 =========
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ========= 图生成配置 =========
GRAPH_N_NODES = 8                # 固定顶点数(MLP要求输入维度固定)
GRAPH_EDGE_DENSITY = 0.35        # 边密度 0~1
GRAPH_CAP_MIN = 1
GRAPH_CAP_MAX = 20

# ========= 数据集配置 =========
DATASET_TRAIN_SAMPLES = 6000
DATASET_TEST_SAMPLES = 1500

# ========= 神经网络MLP配置 =========
NN_HIDDEN_SIZES = [128, 64, 32]  # 隐藏层维度
NN_LEARNING_RATE = 1e-4
NN_EPOCHS = 80
NN_BATCH_SIZE = 128

# ========= 设备自动选择(M2Max MPS / CUDA / CPU) =========
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# MPS不支持float64，强制float32
TORCH_DTYPE = torch.float32

# 模型保存路径
MODEL_SAVE_PATH = OUTPUT_DIR / "mlp_maxflow.pth"
TRAIN_LOG_CSV = OUTPUT_DIR / "train_log.csv"
TEST_RESULT_CSV = OUTPUT_DIR / "test_result.csv"
DATASET_NPZ = OUTPUT_DIR / "dataset.npz"
# 绘图输出
LOSS_PLOT_PNG = OUTPUT_DIR / "loss_plot.png"
SAMPLE_GRAPH_PNG = OUTPUT_DIR / "sample_flow_graph.png"
SINGLE_COMPARE_JSON = OUTPUT_DIR / "single_compare.json"
