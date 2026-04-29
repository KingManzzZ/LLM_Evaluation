# LLM_Evaluation

这是一个关于大语言模型 (LLM) 评测的项目，旨在通过多种维度（公平性、可靠性、安全性、逻辑推理等）对模型性能进行评估。

## 项目结构

```text
QA/
├── Assess/                         # 评测核心代码
│   ├── assess_fairness/            # 公平性评测 (偏见、歧视等)
│   ├── assess_reliability/         # 可靠性评测
│   ├── assess_security/            # 安全性评测 (诱导、攻击测试等)
│   ├── complex_reasoning/          # 复杂推理能力 (数学、逻辑、因果)
│   ├── long_text_comprehension/    # 长文本理解能力
│   └── token_and_throughput/       # 性能测试 (Token 生成速率、吞吐量)
├── dataset/                        # 各维度评测的数据集 (Git 会忽略，需自行准备)
├── largemodel_create_and_evaluate/ # 自动化问题生成与演化脚本
├── model/                          # 评测过程中使用到的特定判定模型或接口
├── narrativeqa/                    # 针对叙事性问答的数据处理与分析
├── app.py                          # 应用程序主入口或 API 服务
├── config.py                       # 项目全局配置文件
└── requirements.txt                # 运行所需的 Python 依赖
```

## 主要功能

- **多维度评测**：涵盖安全性、公平性、可靠性、推理能力、长文本理解等核心模型指标。
- **自动评估**：利用专门的判定模型（如 `model/math_judge_model.py`）对模型输出进行自动打分和评估。
- **题目生成与演化**：支持通过 `largemodel_create_and_evaluate` 模块自动生成评估问题并进行题目演化（Questions Evolving）。
- **性能分析**：测量不同模型的推理效率和资源消耗。

## 快速开始

### 1. 环境准备

建议使用 Python 3.10+ 版本，并创建虚拟环境：

```powershell
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境 (Windows)
.\venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

在根目录下根据需要配置 `.env` 文件或修改 `config.py`，设置模型 API 密钥或相关路径。

### 3. 运行评测

您可以根据需要运行特定的子模块，例如：

```powershell
# 运行安全性评测
python Assess/assess_security/main.py
```

## 注意事项

- **数据集**：`dataset/` 目录下的具体数据文件已通过 `.gitignore` 排除，请在使用前确保数据路径正确。
- **模型缓存**：`ModelCache/` 目录用于存储下载的模型权重，已被忽略以减小仓库体积。

## 许可证

[MIT License](LICENSE) (根据实际情况调整)

