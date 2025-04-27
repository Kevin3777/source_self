import os
import json
import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import Counter

def load_test_data(file_path):
    """从JSON文件加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_category_prediction(
    model_path,
    test_data,
    output_file=None,
    max_new_tokens=50,
    temperature=0.7,
    num_samples=None,  # 可选参数，用于限制评估的样本数
    device=None,       # 添加设备参数，允许指定运行设备
    batch_size=1,      # 批处理大小，加快处理速度
    interval=200,      # 中间结果统计间隔
    display_examples=True  # 是否显示生成样例
):
    """
    评估end_url模型的类别预测能力
    
    Args:
        model_path: 模型路径
        test_data: 测试数据（JSON列表或文件路径）
        output_file: 输出结果文件路径（可选）
        max_new_tokens: 生成的最大token数
        temperature: 生成温度
        num_samples: 要评估的样本数（如果为None，则评估所有样本）
        device: 运行设备 (None会自动选择可用设备)
        batch_size: 批处理大小（默认为1）
        interval: 中间结果统计间隔（默认每200个样本统计一次）
        display_examples: 是否在控制台显示生成结果样例
        
    Returns:
        tuple: (结果列表, 评估摘要)
    """
    print(f"评估end_url模型（类别预测）: {model_path}")
    
    # 加载测试数据（如果是文件路径）
    if isinstance(test_data, str):
        test_data = load_test_data(test_data)
    
    # 如果指定了样本数，则限制测试样本
    if num_samples is not None and num_samples < len(test_data):
        import random
        random.shuffle(test_data)
        test_data = test_data[:num_samples]
    
    # 确定设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 确保pad_token存在，否则设置为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 显式指定模型设备
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    
    # 增加生成参数，加快生成速度
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        # 增加以下参数来加快生成速度
        "num_beams": 1,  # 禁用束搜索
        "early_stopping": True
    }
    
    # 打印确认信息
    print(f"模型已加载到设备: {next(model.parameters()).device}")
    print(f"样本总数: {len(test_data)}")
    print(f"批处理大小: {batch_size}")
    print(f"中间结果统计间隔: {interval}个样本")
    
    results = []
    category_correct = 0
    subcategory_correct = 0
    total = 0
    
    # 收集所有类别和预测，用于计算混淆矩阵和详细分析
    all_true_categories = []
    all_true_subcategories = []
    all_predicted_categories = []
    all_predicted_subcategories = []
    
    start_time = time.time()
    
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i+batch_size]
        batch_results = []
        
        for item in batch_data:
            # 提取必要信息
            true_abstract = item["text"].strip()
            categories = item["categories"]
            main_category = categories[0]  # 例如："cs.CV"
            
            # 将主类别分为类别和子类别
            parts = main_category.split('.')
            true_category = parts[0]  # 例如："cs"
            true_subcategory = parts[1] if len(parts) > 1 else ""  # 例如："CV"
            
            all_true_categories.append(true_category)
            all_true_subcategories.append(true_subcategory)
            
            # 创建输入提示
            prompt = f"[Abstract: {true_abstract}] <src>"
            
            # 生成文本
            inputs = tokenizer(prompt, return_tensors="pt")
            # 确保输入张量在正确的设备上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 添加attention_mask如果不存在
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # 使用采样生成
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_config
                    )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # 从输出中提取预测的类别
                # 假设格式为："[Abstract: ...] <src> cs.xxx.xx.id </src>"
                category_match = re.search(r'<src>\s*([^<]+)\s*</src>', generated_text)
                if category_match:
                    predicted_full = category_match.group(1).strip()
                    # 仅提取类别部分（例如，从"cs.CV.2503.10638v1"中提取"cs.CV"）
                    category_parts = predicted_full.split('.')
                    if len(category_parts) >= 2:
                        predicted_category = category_parts[0]
                        predicted_subcategory = category_parts[1]
                    else:
                        predicted_category = predicted_full
                        predicted_subcategory = ""
                else:
                    predicted_category = ""
                    predicted_subcategory = ""
                    predicted_full = ""
                
                all_predicted_categories.append(predicted_category)
                all_predicted_subcategories.append(predicted_subcategory)
                
                # 检查预测是否正确
                cat_correct = predicted_category.lower() == true_category.lower()
                subcat_correct = predicted_subcategory.lower() == true_subcategory.lower()
                
                if cat_correct:
                    category_correct += 1
                if subcat_correct:
                    subcategory_correct += 1
                
                total += 1
                
                result = {
                    "true_abstract": true_abstract[:200] + "...",  # 仅保存摘要的开头部分以减小文件大小
                    "true_category": true_category,
                    "true_subcategory": true_subcategory,
                    "predicted_full": predicted_full,
                    "predicted_category": predicted_category,
                    "predicted_subcategory": predicted_subcategory,
                    "category_correct": cat_correct,
                    "subcategory_correct": subcat_correct,
                }
                
                batch_results.append(result)
                
                # 显示生成的类别和真实类别
                if display_examples and len(results) < 2:  # 只显示前两个样本的详细信息
                    print("\n" + "="*50)
                    print(f"样本 {len(results)+1}:")
                    print("\n真实摘要片段:")
                    print(true_abstract[:150] + "...")
                    print(f"\n真实类别: {true_category}.{true_subcategory}")
                    print(f"预测类别: {predicted_category}.{predicted_subcategory}")
                    print(f"类别正确: {cat_correct}, 子类别正确: {subcat_correct}")
                    print("="*50)
                
            except Exception as e:
                print(f"处理样本 {i+1} 时出错: {e}")
                continue
        
        # 添加批处理结果
        results.extend(batch_results)
        
        # 显示进度和当前平均得分
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or current_sample_count == len(test_data) or i + batch_size >= len(test_data):
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            
            # 计算当前准确率
            current_category_accuracy = category_correct / total if total > 0 else 0
            current_subcategory_accuracy = subcategory_correct / total if total > 0 else 0
            
            print(f"\n[进度] 已处理: {current_sample_count}/{len(test_data)} 样本 " +
                  f"({current_sample_count/len(test_data)*100:.1f}%) | " +
                  f"速度: {samples_per_second:.2f} 样本/秒 | " +
                  f"剩余时间: {(len(test_data)-current_sample_count)/samples_per_second/60:.1f} 分钟")
            
            print(f"[中间结果] 类别准确率: {current_category_accuracy:.4f} | " +
                  f"子类别准确率: {current_subcategory_accuracy:.4f}")
    
    # 计算准确率
    category_accuracy = category_correct / total if total > 0 else 0
    subcategory_accuracy = subcategory_correct / total if total > 0 else 0
    
    # 计算类别和子类别的分布
    true_category_counts = Counter(all_true_categories)
    true_subcategory_counts = Counter(all_true_subcategories)
    predicted_category_counts = Counter(all_predicted_categories)
    predicted_subcategory_counts = Counter(all_predicted_subcategories)
    
    elapsed_time = time.time() - start_time
    
    # 准备结果摘要
    summary = {
        "category_accuracy": category_accuracy,
        "subcategory_accuracy": subcategory_accuracy,
        "total_samples": total,
        "evaluation_time": elapsed_time,
        "samples_per_second": total/elapsed_time,
        "category_distribution": {
            "true": dict(true_category_counts),
            "predicted": dict(predicted_category_counts)
        },
        "subcategory_distribution": {
            "true": dict(true_subcategory_counts),
            "predicted": dict(predicted_subcategory_counts)
        }
    }
    
    # 输出结果摘要
    print("\n" + "="*50)
    print("==== 类别预测评估结果 ====")
    print(f"评估样本数: {total}")
    print(f"总用时: {elapsed_time:.2f} 秒 (平均 {total/elapsed_time:.2f} 样本/秒)")
    print(f"类别准确率: {category_accuracy:.4f}")
    print(f"子类别准确率: {subcategory_accuracy:.4f}")
    
    # 打印最常见的类别
    print("\n最常见的真实类别:")
    for cat, count in true_category_counts.most_common(5):
        print(f"  {cat}: {count} 个样本")
    
    print("\n最常见的真实子类别:")
    for subcat, count in true_subcategory_counts.most_common(5):
        print(f"  {subcat}: {count} 个样本")
    
    print("\n最常见的预测类别:")
    for cat, count in predicted_category_counts.most_common(5):
        print(f"  {cat}: {count} 个样本")
        
    print("\n最常见的预测子类别:")
    for subcat, count in predicted_subcategory_counts.most_common(5):
        print(f"  {subcat}: {count} 个样本")
    
    print("="*50)
    
    # 保存详细结果
    if output_file:
        output = {
            "model_path": model_path,
            "evaluation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "num_samples": len(results)
            },
            "summary": summary,
            "individual_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"详细结果已保存到: {output_file}")
    
    return results, summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估end_url模型的类别预测能力")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="训练好的end_url模型路径")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="测试数据JSON文件路径")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="输出结果文件路径")
    parser.add_argument("--max_tokens", type=int, default=50, 
                        help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="生成温度")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="要评估的样本数（默认为全部）")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="运行设备 (默认自动选择)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小（加快评估）")
    parser.add_argument("--interval", type=int, default=200,
                        help="中间结果统计间隔（每X个样本统计一次）")
    parser.add_argument("--no_display", action="store_true",
                        help="不显示生成样例")
    
    args = parser.parse_args()
    
    # 评估模型
    evaluate_category_prediction(
        model_path=args.model_path,
        test_data=args.test_data,
        output_file=args.output_file,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        device=args.device,
        batch_size=args.batch_size,
        interval=args.interval,
        display_examples=not args.no_display
    )