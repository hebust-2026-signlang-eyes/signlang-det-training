# signlang-det-training

[English](README.md) | [简体中文](README.zh-CN.md)

一个适用于在 Kaggle 中训练通用动态手语特征编码器的开源项目，完整流程封装在可独立运行的 Notebook 中。

## 项目简介 🚀

本项目用于训练 `signlang_det`：一个面向原型检索的轻量手部时序编码器，完整流程面向 Kaggle GPU Session 设计。编码器先从 Google Isolated Sign Language Recognition 语料中学习通用时序表示，再使用目标 landmark 数据进行领域适配。

训练期间使用的分类头只提供辅助监督，不会包含在最终导出的编码器中，也不会限制推理阶段可使用的手语词表。

项目中使用的识别思路思路深受 [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) 竞赛第一名作品 [https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with]() 启发。

完整流程见 [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb)。

项目也支持交付带 Tag 的版本：Kaggle CD 会把排队的 Tag 依次提交到同一个稳定 Kaggle Notebook；训练完成后，由仓库写入者通过锁定的 Issue 上传 ZIP。Issue workflow 会按顺序验证并转换候选包，再发布对应的 GitHub Release。配置和故障恢复方法见 [Kaggle 持续交付说明](docs/KAGGLE_CD.md)。

## 核心特点 ✨

- 训练、评估和推理统一使用 `hand168-temporal` 预处理
- 采用腕点相对归一化，并显式处理左右手缺失情况
- 左右手共享 TCN 分支，随后通过带掩码的 Transformer 融合
- 联合使用交叉熵、监督对比损失和 batch-hard triplet 损失
- 使用类别完全隔离的验证集和测试集评估开放词表能力
- 先进行池化余弦检索，再使用受限帧级 DTW 重排
- 支持可恢复的 memory-mapped 缓存和原子 checkpoint
- 导出带 SHA-256 指纹的独立编码器

## 训练流程 🧭

Notebook 从上到下依次完成完整流程：

1. 检查 Kaggle CUDA 环境和所选 GPU 架构。
2. 将 Google ASL parquet 序列转换为可恢复的 `hand168-temporal` 缓存。
3. 使用 P×K 采样训练通用手部时序表示。
4. 在训练时未出现的类别上选择并评估编码器。
5. 审计并预处理目标 object-NPY landmark 数据。
6. 仅加载选中的编码器，重新创建辅助分类头并完成领域适配。
7. 评估池化检索和 Top-K DTW 重排。
8. 导出最终编码器、配置、指标、数据拆分、日志和指纹。

### 特征契约

| 项目 | 约束 |
|---|---|
| 输入 | `float32`，`B × 64 × 168` |
| 每帧特征 | 2 只手 × 21 个关键点 × `(x, y, z, velocity)` |
| Padding 值 | `-100.0` |
| 编码器输出 | `float32`，`B × 64 × 128` |
| 有效帧输出 | L2 归一化 |
| Padding 帧输出 | 全零 |

## 数据要求 📚

运行前需要在 Kaggle Notebook 中挂载以下两个数据集。

### 源语料

[Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) 竞赛数据集需要包含：

- `train.csv`
- `sign_to_prediction_index_map.json`
- `train.csv` 的 `path` 列所引用的 parquet 文件

训练只使用 `left_hand` 和 `right_hand` landmark，不使用脸部和身体 landmark。

### 目标 landmark

目标数据来自 [ASL-preprocessing 7 output](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output)，需要提供一个包含 object-NPY 文件的 `landmarks` 目录。每个有效文件必须包含：

- 字符串标签；
- 形状为 `T × 100 × 3` 的 `float32` landmark 数组。

默认布局使用索引 `0:21` 作为右手、`21:42` 作为左手。如果所挂载数据集使用其他已知布局，请修改 `target_right_slice` 和 `target_left_slice`。Notebook 使用受限 pickle 白名单加载对象文件，并将无效文件记录到拒绝清单。

## 在 Kaggle 上运行 ▶️

1. 将 [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) 导入或上传到 Kaggle。
2. 挂载源语料和目标 landmark 数据集。
3. 选择**单张 T4 GPU** 加速器。
4. 检查配置单元。Notebook 会验证 Competition 和 Notebook Output 的标准挂载路径；使用非标准挂载位置时请显式指定路径。
5. 将 `SMOKE_TEST = True` 可执行短流程集成验证；正式训练时保持 `False`。
6. 选择 **Run all**，并完整保留 `/kaggle/working/signlang-det` 输出目录。

Notebook 默认设置 `num_workers = 0` 并关闭 DataParallel，以提高 Notebook 重启稳定性并避免轻量编码器的额外同步开销。兼容的 checkpoint 和未完成的源数据缓存会自动恢复。

## 配置说明 ⚙️

主要设置位于 Notebook 开头的配置单元中。

| 配置项 | 默认值 | 作用 |
|---|---:|---|
| `source_root` | Kaggle 标准挂载路径 | 可选的 Google ASL 数据集根目录覆盖值 |
| `target_landmarks` | Kaggle 标准挂载路径 | 可选的目标 `landmarks` 目录覆盖值 |
| `work_dir` | `/kaggle/working/signlang-det` | 缓存、checkpoint、指标和导出目录 |
| `max_frames` | `64` | 最大编码序列长度 |
| `min_frames` | `12` | 与运行时动作分段器一致的最短动作长度 |
| `max_input_frames` | `120` | 拒绝样本前允许的最大动作长度 |
| `embedding_dim` | `128` | 帧级和池化特征维度 |
| `top_k` | `20` | 进入 DTW 重排的池化候选数量 |
| `dtw_window` | `12` | 受限 DTW 窗口 |
| `resume` | `True` | 恢复兼容的训练状态 |
| `use_data_parallel` | `False` | 可选的双 GPU 执行路径 |

正式配置使用 30 epoch × 300 batch 进行通用表示训练，并使用 40 epoch × 200 batch 进行领域适配。这里的 epoch 由固定数量的 P×K batch 组成，不代表完整遍历数据集。

## 输出产物 📦

Notebook 验证最终编码器后会删除缓存、checkpoint、状态文件和其它中间数据。`/kaggle/working/signlang-det` 最终只包含：

- `signlang_det_encoder.pt`，包含最终编码器契约、权重和指纹；
- `figures/training_curves.png` 和 `figures/retrieval_summary.png`；
- `representation_training/metrics.csv` 和 `domain_adaptation/metrics.csv`；
- `representation_training/train.log` 和 `domain_adaptation/train.log`。

Kaggle CD 会将 PT 转为 ONNX 和 RKNN。三个模型、`model-manifest.json` 及 Tag 对应的原始 Notebook 分别作为独立 Release 资产，其余 Notebook 输出统一打包为 `notebook-output.tar.gz`。
每个动态原型都必须同时保存编码器指纹和 `hand168-temporal` 预处理标识。由其他编码器或预处理契约生成的原型不能与当前导出模型混用。

## 限制与校准 ⚠️

- 完整训练依赖 Kaggle 挂载数据集和兼容的 CUDA 环境；仓库本地环境缺少这些条件时无法复现正式训练。
- 当前流程描述的 PyTorch CUDA 构建以 T4 为目标。仅检测到 CUDA 可用并不代表 GPU 架构兼容。
- Notebook 不包含动作分段、服务端 API 或原型管理界面。
- Unknown 手语的距离阈值和 margin 阈值不能从训练 loss 推导，必须使用同时包含已知查询、困难负样本和未知动作的独立数据进行校准。
- 各训练数据集的许可证和使用限制由其所有者负责；本仓库不重新分发任何训练数据。

## 鸣谢 🙏

- 感谢 [hoyso48](https://www.kaggle.com/hoyso48) 在 [https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with]() 中提供的宝贵的思路。
- 感谢 [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) 竞赛的组织者和贡献者通过 Kaggle 提供源语料。
- 感谢 [Abdelrhman Karam](https://www.kaggle.com/abdelrhmankaram) 发布用于目标领域适配的 [ASL-preprocessing 7 output](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output)。

所有数据仍受其原始许可证、竞赛规则和使用条款约束。

## 项目文件 🗂️

| 文件 | 说明 |
|---|---|
| [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) | 自包含的 Kaggle 训练与导出流程 |
| [README.md](README.md) | 英文文档 |
| [docs/KAGGLE_CD.md](docs/KAGGLE_CD.md) | 基于 Tag 的 Kaggle 自动交付配置与运维说明 |
| [LICENSE](LICENSE) | 项目许可证 |

## 许可证 📄

本项目基于 [Apache License 2.0](LICENSE) 开源。
