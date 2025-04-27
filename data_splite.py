import json
import os
import re
from sklearn.model_selection import train_test_split

"""
数据流程与模型训练流程
======================

本文件描述了从原始数据到训练和评估两个不同模型的完整流程。

1. begin_url数据处理 -> begin_url模型训练 -> 摘要生成评估
   输入格式: <src> cs.xxx.xx.id </src> [Abstract: ...]
   任务: 基于论文ID生成摘要
   
2. end_url数据处理 -> end_url模型训练 -> 类别预测评估
   输入格式: [Abstract: ...] <src> cs.xxx.xx.id </src>
   任务: 基于摘要预测论文类别

两种模型是分开训练和评估的，各自针对不同的任务进行优化。
"""

def prepare_datasets(input_file, output_base_dir, test_ratio=0.2, random_seed=42):
    """
    准备两种格式的数据集并分割为训练集和测试集
    
    Args:
        input_file: 原始JSON数据文件
        output_base_dir: 输出目录基础路径
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    Returns:
        dict: 包含所有输出路径的字典
    """
    # 创建必要的目录
    begin_url_dir = os.path.join(output_base_dir, "begin_url")
    end_url_dir = os.path.join(output_base_dir, "end_url")
    
    for dir_path in [begin_url_dir, end_url_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 分割为训练集和测试集
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_seed)
    
    print(f"已将数据分割为训练集({len(train_data)}个样本)和测试集({len(test_data)}个样本)")
    
    # 处理为begin_url格式
    begin_url_train = process_to_begin_url_format(train_data)
    begin_url_test = process_to_begin_url_format(test_data)
    
    # 处理为end_url格式
    end_url_train = process_to_end_url_format(train_data)
    end_url_test = process_to_end_url_format(test_data)
    
    # 保存处理后的数据
    begin_url_train_path = os.path.join(begin_url_dir, "train.txt")
    begin_url_test_path = os.path.join(begin_url_dir, "test.txt")
    end_url_train_path = os.path.join(end_url_dir, "train.txt")
    end_url_test_path = os.path.join(end_url_dir, "test.txt")
    
    save_samples(begin_url_train, begin_url_train_path)
    save_samples(begin_url_test, begin_url_test_path)
    save_samples(end_url_train, end_url_train_path)
    save_samples(end_url_test, end_url_test_path)
    
    # 保存原始JSON测试数据（用于评估）
    begin_url_test_json_path = os.path.join(begin_url_dir, "test_data.json")
    end_url_test_json_path = os.path.join(end_url_dir, "test_data.json")
    
    with open(begin_url_test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    with open(end_url_test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    return {
        "begin_url": {
            "train": begin_url_train_path,
            "test": begin_url_test_path,
            "test_json": begin_url_test_json_path
        },
        "end_url": {
            "train": end_url_train_path,
            "test": end_url_test_path,
            "test_json": end_url_test_json_path
        }
    }

def extract_paper_id(paper_str):
    """从编码的论文字符串中提取论文ID"""
    return paper_str.replace("▁", "/")

def process_to_begin_url_format(data_items):
    """
    处理数据为begin_url格式: <src> cs.xxx.xx.id </src> [Abstract: 摘要内容]
    
    适用于: 基于论文ID生成摘要的任务
    """
    samples = []
    for item in data_items:
        paper_id = extract_paper_id(item["enc-paper-str"])
        abstract = item["text"].strip()
        categories = item["categories"]
        main_category = categories[0]  # 例如："cs.CV"
        
        sample = f"<src> {main_category}.{paper_id} </src> [Abstract: {abstract}]"
        samples.append(sample)
    
    return samples

def process_to_end_url_format(data_items):
    """
    处理数据为end_url格式: [Abstract: 摘要内容] <src> cs.xxx.xx.id </src>
    
    适用于: 基于摘要预测论文类别的任务
    """
    samples = []
    for item in data_items:
        paper_id = extract_paper_id(item["enc-paper-str"])
        abstract = item["text"].strip()
        categories = item["categories"]
        main_category = categories[0]  # 例如："cs.CV"
        
        sample = f"[Abstract: {abstract}] <src> {main_category}.{paper_id} </src>"
        samples.append(sample)
    
    return samples

def save_samples(samples, output_path):
    """将样本保存到文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(samples))
    print(f"已将{len(samples)}个样本保存到 {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备begin_url和end_url格式的数据集")
    parser.add_argument("--input_file", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="dataset", help="输出目录")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    
    args = parser.parse_args()
    
    # 处理数据
    paths = prepare_datasets(
        args.input_file, 
        args.output_dir,
        test_ratio=args.test_size
    )
    
    print("\n数据处理完成!")
    print("begin_url 数据格式: <src> cs.xxx.xx.id </src> [Abstract: 摘要内容]")
    print("end_url 数据格式: [Abstract: 摘要内容] <src> cs.xxx.xx.id </src>")
    print("\n数据路径:")
    print(f"begin_url训练集: {paths['begin_url']['train']}")
    print(f"begin_url测试集: {paths['begin_url']['test']}")
    print(f"end_url训练集: {paths['end_url']['train']}")
    print(f"end_url测试集: {paths['end_url']['test']}")