# README.md（课程设计｜最大流 + 神经网络逼近）
```markdown
# Max‑Flow Project
课程设计项目：基于神经网络逼近求解网络最大流问题

## 项目简介
本项目实现两种求解**最大流问题**的方案：
1. 精确算法：Ford‑Fulkerson（福特‑富尔克森）算法
2. 神经网络逼近方案：使用 MLP 多层感知机对最大流做预测

程序为跨平台桌面GUI应用，支持 Windows / Linux。
网络图由程序随机生成，**不需要外部数据集**。
可以对比精确算法结果与神经网络预测结果，输出误差、运行时间，可视化网络图与训练曲线。

### 功能
- 根据参数随机生成有向流量网络（顶点数量、边密度、边容量可配置）
- Ford‑Fulkerson 精确算法计算真实最大流
- 神经网络模型训练，对最大流做逼近预测
- GUI图形界面操作
- 可视化网络结构图、损失训练曲线
- 输出统计：真实最大流、预测值、误差、程序运行耗时
- 导出对比表格、图像结果

## 环境依赖
Python >=3.9

安装依赖包：
```bash
pip install -r requirements.txt
```

主要库：
- NetworkX — 图结构处理
- PyTorch — 神经网络训练
- PyQt5 — GUI图形界面
- Matplotlib — 绘图可视化

## 运行方式
```bash
python main.py
```

## 项目结构
```
max_flow_project/
├── gui/                # GUI界面模块
├── main.py             # 程序入口
├── graph_generator.py  # 随机网络图生成器
├── edmonds_karp.py     # Edmonds‑Karp算法实现
├── nn_approximator.py  # 神经网络逼近模型
├── visualizer.py       # 可视化绘图模块
├── utils.py            # 工具函数
├── config.py           # 参数配置
├── single_compare.py   # 单组样本对比测试
├── requirements.txt    # 依赖列表
└── README.md
```

>注意：
> `output/` 文件夹为程序运行后自动生成（图片、csv日志）
> `models/` 为训练后保存的模型权重，运行代码本地自动生成，不需要上传仓库。

## 课程信息
>课程设计题目：Разработка приложения для решения задачи о максимальном потоке с нейросетевым подходом
>学生：Мэн Цзияо（蒙继尧）
```

### 配套 requirements.txt 内容，一并复制，网页新建文件命名 `requirements.txt`
```txt
networkx
torch
matplotlib
pyqt5
numpy
```
