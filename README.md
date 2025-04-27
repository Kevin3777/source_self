# CS论文摘要生成与类别预测项目

本项目基于TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T模型，训练了两个独立的模型来处理CS（计算机科学）论文数据：

1. **begin_url模型**：给定论文ID和类别，生成论文摘要
2. **end_url模型**：给定论文摘要，预测论文类别和子类别

## 项目概述

该项目展示了如何利用小型语言模型进行两种不同类型的文本生成任务。通过使用预训练的TinyLlama模型并采用不同的数据格式进行微调，我们可以得到专门针对不同任务优化的模型。

### begin_url模型（摘要生成）

- **输入格式**：`<src> cs.xxx.xx.id </src> [Abstract:`
- **输出**：生成的论文摘要
- **评估指标**：ROUGE-1、ROUGE-2、ROUGE-L

### end_url模型（类别预测）

- **输入格式**：`[Abstract: 摘要内容] <src>`
- **输出**：`<src> cs.xxx.xx.id </src>`（其中cs.xxx为预测的类别和子类别）
- **评估指标**：类别准确率和子类别准确率

## 环境设置

### 安装依赖

```bash
pip install transformers datasets torch rouge numpy scikit-learn
```

注意：使用`scikit-learn`而不是`sklearn`，因为后者已弃用。

### 硬件要求

- 推荐使用GPU进行训练（最低4GB显存）
- 评估可以在CPU上运行，但速度会较慢

## 数据格式

### 输入数据

输入数据应为JSON格式，包含多个论文条目，每个条目具有以下字段：
- `title`: 论文标题
- `text`: 论文摘要
- `enc-paper-str`: 论文ID (格式如 "cs▁AI▁2503.10638v1")
- `categories`: 类别列表 (如 ["cs.CV", "cs.AI", "cs.LG"])

示例:
```json
[
  {
    "title": "Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective",
    "text": "Classifier-free guidance has become a staple for conditional generation with denoising diffusion models...",
    "enc-paper-str": "cs▁AI▁2503.10638v1",
    "categories": ["cs.CV", "cs.AI", "cs.LG"]
  }
]
```

### 处理后的数据格式

数据流程会将原始JSON数据处理成两种不同格式：

1. **begin_url格式**：
   ```
   <src> cs.CV.2503.10638v1 </src> [Abstract: Classifier-free guidance has become a staple for conditional generation...]
   ```

2. **end_url格式**：
   ```
   [Abstract: Classifier-free guidance has become a staple for conditional generation...] <src> cs.CV.2503.10638v1 </src>
   ```

## 使用方法

### 完整工作流程

使用完整工作流脚本可一键完成从数据准备到模型评估的全部步骤：

```bash
python complete_workflow_updated.py --input_file cs_papers.json --output_dir output --epochs 3
```

#### 主要参数

- `--input_file`：输入JSON文件路径（**必需**）
- `--output_dir`：输出目录（默认：output）
- `--test_size`：测试集比例（默认：0.2）
- `--epochs`：训练轮次（默认：3）
- `--batch_size`：批次大小（默认：4）
- `--learning_rate`：学习率（默认：2e-5）
- `--begin_url_max_tokens`：摘要生成的最大token数（默认：200）
- `--end_url_max_tokens`：类别预测的最大token数（默认：50）
- `--eval_samples`：评估时使用的样本数（默认使用全部测试集）

#### 控制流程参数

- `--skip_data_prep`：跳过数据准备步骤
- `--skip_begin_url_training`：跳过begin_url模型训练
- `--skip_end_url_training`：跳过end_url模型训练
- `--skip_begin_url_evaluation`：跳过begin_url模型评估
- `--skip_end_url_evaluation`：跳过end_url模型评估

### 手动执行各步骤

如果需要更精细的控制，可以手动执行各个步骤：

#### 1. 数据准备

```bash
python data_flow_updated.py --input_file cs_papers.json --output_dir dataset
```

这将创建：
- `dataset/begin_url/train.txt`：摘要生成训练数据
- `dataset/begin_url/test.txt`：摘要生成测试数据
- `dataset/begin_url/test_data.json`：用于评估的原始JSON测试数据
- `dataset/end_url/train.txt`：类别预测训练数据
- `dataset/end_url/test.txt`：类别预测测试数据
- `dataset/end_url/test_data.json`：用于评估的原始JSON测试数据

#### 2. 训练模型

训练begin_url模型（摘要生成）：
```bash
python model_training_updated.py --model_type begin_url --train_file dataset/begin_url/train.txt --output_dir models
```

训练end_url模型（类别预测）：
```bash
python model_training_updated.py --model_type end_url --train_file dataset/end_url/train.txt --output_dir models
```

#### 3. 评估模型

评估begin_url模型（摘要生成）：
```bash
python evaluation_abstract_generation.py --model_path models/begin_url/final_model --test_data dataset/begin_url/test_data.json --output_file begin_url_results.json
```

评估end_url模型（类别预测）：
```bash
python evaluation_category_prediction.py --model_path models/end_url/final_model --test_data dataset/end_url/test_data.json --output_
