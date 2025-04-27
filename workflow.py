#!/usr/bin/env python
"""
完整工作流脚本，处理CS论文摘要数据，训练和评估两种不同的模型：
1. begin_url模型：基于论文ID生成摘要
2. end_url模型：基于摘要预测论文类别
"""

import os
import subprocess
import argparse
import time
import json
from datetime import datetime

def run_command(command, description=None):
    """运行命令并输出结果"""
    if description:
        print(f"\n{'='*10} {description} {'='*10}")
    print(f"运行命令: {command}")
    start_time = time.time()
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"命令执行出错: {command}")
        print(f"错误信息: {result.stderr}")
        return False
    
    print(result.stdout)
    duration = time.time() - start_time
    print(f"完成用时: {duration:.2f} 秒")
    return True

def check_file_exists(file_path, error_message=None):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        if error_message:
            print(error_message)
        else:
            print(f"错误: 文件 {file_path} 不存在!")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="TinyLlama模型训练与评估的完整工作流")
    
    # 输入和输出参数
    parser.add_argument("--input_file", type=str, required=True, 
                        help="包含论文数据的输入JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="所有输出的基础目录")
    
    # 数据相关参数
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="用于测试的数据比例")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=3, 
                        help="训练轮次数")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="学习率")
    parser.add_argument("--begin_url_max_tokens", type=int, default=200,
                        help="begin_url模型生成的最大token数")
    parser.add_argument("--end_url_max_tokens", type=int, default=50,
                        help="end_url模型生成的最大token数")
    
    # 控制流程的参数
    parser.add_argument("--skip_data_prep", action="store_true", 
                        help="跳过数据准备步骤")
    parser.add_argument("--skip_begin_url_training", action="store_true", 
                        help="跳过begin_url模型训练")
    parser.add_argument("--skip_end_url_training", action="store_true", 
                        help="跳过end_url模型训练")
    parser.add_argument("--skip_begin_url_evaluation", action="store_true", 
                        help="跳过begin_url模型评估")
    parser.add_argument("--skip_end_url_evaluation", action="store_true", 
                        help="跳过end_url模型评估")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="评估时使用的样本数（默认使用全部测试集）")
    
    args = parser.parse_args()
    
    # 创建时间戳以标识唯一运行ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 定义路径
    dataset_dir = os.path.join(run_dir, "dataset")
    models_dir = os.path.join(run_dir, "models")
    results_dir = os.path.join(run_dir, "results")
    
    # 创建所需目录
    for directory in [dataset_dir, models_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 定义日志文件
    log_file = os.path.join(run_dir, "workflow.log")
    
    # 1. 数据准备
    if not args.skip_data_prep:
        if not check_file_exists(args.input_file, f"错误: 输入文件 {args.input_file} 不存在!"):
            return
            
        data_prep_cmd = (
            f"python data_flow_updated.py --input_file {args.input_file} "
            f"--output_dir {dataset_dir} --test_size {args.test_size}"
        )
        if not run_command(data_prep_cmd, "步骤1: 数据准备"):
            print("数据准备失败，终止流程。")
            return
    else:
        print("\n跳过步骤1: 数据准备")
    
    # 定义数据路径
    begin_url_train_path = os.path.join(dataset_dir, "begin_url", "train.txt")
    begin_url_test_path = os.path.join(dataset_dir, "begin_url", "test.txt")
    begin_url_test_json_path = os.path.join(dataset_dir, "begin_url", "test_data.json")
    
    end_url_train_path = os.path.join(dataset_dir, "end_url", "train.txt")
    end_url_test_path = os.path.join(dataset_dir, "end_url", "test.txt")
    end_url_test_json_path = os.path.join(dataset_dir, "end_url", "test_data.json")
    
    # 确认数据文件存在
    required_files = [
        (begin_url_train_path, "begin_url训练数据"),
        (begin_url_test_json_path, "begin_url测试数据JSON"),
        (end_url_train_path, "end_url训练数据"),
        (end_url_test_json_path, "end_url测试数据JSON")
    ]
    
    for file_path, file_desc in required_files:
        if not args.skip_data_prep and not check_file_exists(file_path, f"错误: {file_desc}文件 {file_path} 不存在!"):
            return
    
    # 2. 训练begin_url模型（摘要生成）
    if not args.skip_begin_url_training:
        begin_url_train_cmd = (
            f"python model_training_updated.py --model_type begin_url "
            f"--train_file {begin_url_train_path} --output_dir {models_dir} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.learning_rate}"
        )
        if not run_command(begin_url_train_cmd, "步骤2: 训练begin_url模型（摘要生成）"):
            print("begin_url模型训练失败，但将继续流程。")
    else:
        print("\n跳过步骤2: 训练begin_url模型")
    
    # 3. 训练end_url模型（类别预测）
    if not args.skip_end_url_training:
        end_url_train_cmd = (
            f"python model_training_updated.py --model_type end_url "
            f"--train_file {end_url_train_path} --output_dir {models_dir} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.learning_rate}"
        )
        if not run_command(end_url_train_cmd, "步骤3: 训练end_url模型（类别预测）"):
            print("end_url模型训练失败，但将继续流程。")
    else:
        print("\n跳过步骤3: 训练end_url模型")
    
    # 定义模型路径
    begin_url_model_path = os.path.join(models_dir, "begin_url", "final_model")
    end_url_model_path = os.path.join(models_dir, "end_url", "final_model")
    
    # 4. 评估begin_url模型（摘要生成）
    if not args.skip_begin_url_evaluation:
        if os.path.exists(begin_url_model_path):
            eval_samples_param = f"--num_samples {args.eval_samples}" if args.eval_samples else ""
            begin_url_eval_cmd = (
                f"python evaluation_abstract_generation.py --model_path {begin_url_model_path} "
                f"--test_data {begin_url_test_json_path} "
                f"--output_file {os.path.join(results_dir, 'begin_url_results.json')} "
                f"--max_tokens {args.begin_url_max_tokens} {eval_samples_param}"
            )
            if not run_command(begin_url_eval_cmd, "步骤4: 评估begin_url模型（摘要生成）"):
                print("begin_url模型评估失败，但将继续流程。")
        else:
            print(f"\n跳过步骤4: begin_url模型不存在于 {begin_url_model_path}")
    else:
        print("\n跳过步骤4: 评估begin_url模型")
    
    # 5. 评估end_url模型（类别预测）
    if not args.skip_end_url_evaluation:
        if os.path.exists(end_url_model_path):
            eval_samples_param = f"--num_samples {args.eval_samples}" if args.eval_samples else ""
            end_url_eval_cmd = (
                f"python evaluation_category_prediction.py --model_path {end_url_model_path} "
                f"--test_data {end_url_test_json_path} "
                f"--output_file {os.path.join(results_dir, 'end_url_results.json')} "
                f"--max_tokens {args.end_url_max_tokens} {eval_samples_param}"
            )
            if not run_command(end_url_eval_cmd, "步骤5: 评估end_url模型（类别预测）"):
                print("end_url模型评估失败，但将继续流程。")
        else:
            print(f"\n跳过步骤5: end_url模型不存在于 {end_url_model_path}")
    else:
        print("\n跳过步骤5: 评估end_url模型")
    
    # 6. 生成综合报告
    print("\n" + "="*50)
    print(f"工作流程已完成。所有输出保存在: {run_dir}")
    
    # 加载评估结果（如果存在）
    begin_url_results_path = os.path.join(results_dir, 'begin_url_results.json')
    end_url_results_path = os.path.join(results_dir, 'end_url_results.json')
    
    summary = {
        "run_id": timestamp,
        "timestamp": datetime.now().isoformat(),
        "input_file": args.input_file,
        "output_directory": run_dir,
        "parameters": {
            "test_size": args.test_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "begin_url_max_tokens": args.begin_url_max_tokens,
            "end_url_max_tokens": args.end_url_max_tokens,
            "eval_samples": args.eval_samples
        },
        "models": {
            "begin_url": {
                "trained": not args.skip_begin_url_training and os.path.exists(begin_url_model_path),
                "evaluated": not args.skip_begin_url_evaluation and os.path.exists(begin_url_results_path),
                "model_path": begin_url_model_path if os.path.exists(begin_url_model_path) else None,
                "results_path": begin_url_results_path if os.path.exists(begin_url_results_path) else None
            },
            "end_url": {
                "trained": not args.skip_end_url_training and os.path.exists(end_url_model_path),
                "evaluated": not args.skip_end_url_evaluation and os.path.exists(end_url_results_path),
                "model_path": end_url_model_path if os.path.exists(end_url_model_path) else None,
                "results_path": end_url_results_path if os.path.exists(end_url_results_path) else None
            }
        }
    }
    
    # 添加评估结果摘要（如果有）
    if os.path.exists(begin_url_results_path):
        try:
            with open(begin_url_results_path, 'r', encoding='utf-8') as f:
                begin_url_data = json.load(f)
                summary["models"]["begin_url"]["results_summary"] = begin_url_data.get("average_scores", {})
        except Exception as e:
            print(f"无法读取begin_url结果: {e}")
    
    if os.path.exists(end_url_results_path):
        try:
            with open(end_url_results_path, 'r', encoding='utf-8') as f:
                end_url_data = json.load(f)
                if "summary" in end_url_data:
                    summary["models"]["end_url"]["results_summary"] = {
                        "category_accuracy": end_url_data["summary"].get("category_accuracy", 0),
                        "subcategory_accuracy": end_url_data["summary"].get("subcategory_accuracy", 0)
                    }
        except Exception as e:
            print(f"无法读取end_url结果: {e}")
    
    # 保存综合报告
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"运行摘要已保存到: {summary_path}")
    print("="*50)
    
    # 打印结果摘要
    if "results_summary" in summary["models"]["begin_url"]:
        rouge_scores = summary["models"]["begin_url"]["results_summary"]
        print("\nbegin_url模型（摘要生成）结果:")
        print(f"  ROUGE-1: {rouge_scores.get('rouge-1', 0):.4f}")
        print(f"  ROUGE-2: {rouge_scores.get('rouge-2', 0):.4f}")
        print(f"  ROUGE-L: {rouge_scores.get('rouge-l', 0):.4f}")
    
    if "results_summary" in summary["models"]["end_url"]:
        accuracy = summary["models"]["end_url"]["results_summary"]
        print("\nend_url模型（类别预测）结果:")
        print(f"  类别准确率: {accuracy.get('category_accuracy', 0):.4f}")
        print(f"  子类别准确率: {accuracy.get('subcategory_accuracy', 0):.4f}")

if __name__ == "__main__":
    main()