"""项目入口 main()"""
import numpy as np
import torch

from visualizer import plot_loss_curve, draw_flow_network
from single_compare import compare_single_graph
from graph_generator import generate_dataset, generate_flow_graph
from edmonds_karp import edmonds_karp_max_flow
from nn_approximator import MaxFlowMLP, build_dataloader, train_mlp, predict_mlp
from utils import calc_metrics, write_csv, save_npz, save_json

from config import (
    GRAPH_N_NODES, GRAPH_EDGE_DENSITY, GRAPH_CAP_MIN, GRAPH_CAP_MAX,
    DATASET_TRAIN_SAMPLES, DATASET_TEST_SAMPLES,
    NN_HIDDEN_SIZES, NN_LEARNING_RATE, NN_EPOCHS, NN_BATCH_SIZE,
    DEVICE, MODEL_SAVE_PATH, TRAIN_LOG_CSV, TEST_RESULT_CSV, DATASET_NPZ,
    LOSS_PLOT_PNG, SAMPLE_GRAPH_PNG, SINGLE_COMPARE_JSON, OUTPUT_DIR
)


def main():
    print(f"==== Project start. Device = {DEVICE} ====")
    n_nodes = GRAPH_N_NODES
    input_dim = n_nodes * n_nodes

    # 1.生成训练集
    print("\n==== Generate train dataset ====")
    X_train, y_train = generate_dataset(
        n_samples=DATASET_TRAIN_SAMPLES,
        n_nodes=n_nodes,
        edge_density=GRAPH_EDGE_DENSITY,
        cap_min=GRAPH_CAP_MIN,
        cap_max=GRAPH_CAP_MAX,
        calc_maxflow_func=edmonds_karp_max_flow
    )

    # 2.生成测试集
    print("\n==== Generate test dataset ====")
    X_test, y_test = generate_dataset(
        n_samples=DATASET_TEST_SAMPLES,
        n_nodes=n_nodes,
        edge_density=GRAPH_EDGE_DENSITY,
        cap_min=GRAPH_CAP_MIN,
        cap_max=GRAPH_CAP_MAX,
        calc_maxflow_func=edmonds_karp_max_flow
    )

    # 保存数据集缓存
    save_npz(DATASET_NPZ, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Dataset saved -> {DATASET_NPZ}")

    # 3.构建dataloader
    train_loader = build_dataloader(X_train, y_train, batch_size=NN_BATCH_SIZE, shuffle=True)
    test_loader = build_dataloader(X_test, y_test, batch_size=NN_BATCH_SIZE, shuffle=False)

    # 4.初始化模型
    model = MaxFlowMLP(input_dim=input_dim, hidden_sizes=NN_HIDDEN_SIZES).to(DEVICE)
    print(model)

    # 5.训练
    print("\n==== Start training ====")
    log_rows = train_mlp(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=NN_LEARNING_RATE,
        epochs=NN_EPOCHS
    )

    # 保存训练日志csv
    write_csv(TRAIN_LOG_CSV, header=["epoch", "train_loss", "test_loss"], rows=log_rows)

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved -> {MODEL_SAVE_PATH}")

    # 6.测试集推理评估
    y_pred = predict_mlp(model, X_test)
    metrics = calc_metrics(y_test, y_pred)
    print("\n==== Test set metrics ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 保存测试集真值与预测
    test_rows = np.stack([y_test, y_pred], axis=1).tolist()
    write_csv(TEST_RESULT_CSV, header=["y_true_maxflow", "y_pred_mlp"], rows=test_rows)
    print(f"Test result saved -> {TEST_RESULT_CSV}")

    # ====== 绘图：loss曲线、样例网络图 ======
    plot_loss_curve(TRAIN_LOG_CSV, LOSS_PLOT_PNG)

    sample_G, _cap_matrix = generate_flow_graph(
        n_nodes=GRAPH_N_NODES,
        edge_density=GRAPH_EDGE_DENSITY,
        cap_min=GRAPH_CAP_MIN,
        cap_max=GRAPH_CAP_MAX
    )
    draw_flow_network(sample_G, SAMPLE_GRAPH_PNG)

    # ========== 单张图算法对比演示（精确算法 vs NN） ==========
    sample_compare_result = compare_single_graph(sample_G, _cap_matrix, model)
    print("\n==== Single graph compare(Exact VS NN) ====")
    for k, v in sample_compare_result.items():
        print(f"{k}: {v}")

    save_json(sample_compare_result, SINGLE_COMPARE_JSON)
    print(f"Single compare result saved -> {SINGLE_COMPARE_JSON}")

    print("\n==== Pipeline finished. All outputs in ./output ====")


if __name__ == "__main__":
    main()
