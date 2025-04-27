import os
import json
import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge import Rouge
import numpy as np
from tqdm import tqdm

def load_test_data(file_path):
    """从JSON文件加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_paper_id(paper_str):
    """从编码的论文字符串中提取论文ID"""
    return paper_str.replace("▁", "/")

def evaluate_abstract_generation(
    model_path,
    test_data,
    output_file=None,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    num_samples=None,  # 可选参数，用于限制评估的样本数
    device=None,       # 添加设备参数，允许指定运行设备
    batch_size=1,      # 批处理大小，加快处理速度
    interval=200,      # 中间结果统计间隔
    display_examples=True  # 是否显示生成样例
):
    """
    评估begin_url模型的摘要生成能力
    
    Args:
        model_path: 模型路径
        test_data: 测试数据（JSON列表或文件路径）
        output_file: 输出结果文件路径（可选）
        max_new_tokens: 生成的最大token数
        temperature: 生成温度
        top_p: top-p采样参数
        num_samples: 要评估的样本数（如果为None，则评估所有样本）
        device: 运行设备 (None会自动选择可用设备)
        batch_size: 批处理大小（默认为1）
        interval: 中间结果统计间隔（默认每200个样本统计一次）
        display_examples: 是否在控制台显示生成结果样例
        
    Returns:
        tuple: (结果列表, 平均ROUGE得分)
    """
    print(f"评估begin_url模型（摘要生成）: {model_path}")
    
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
        "top_p": top_p,
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
    
    # 初始化Rouge用于评估
    rouge = Rouge()
    
    results = []
    rouge_scores = []
    start_time = time.time()
    
    # 创建进度条
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i+batch_size]
        batch_results = []
        
        for item in batch_data:
            # 提取必要信息
            paper_id = extract_paper_id(item["enc-paper-str"])
            true_abstract = item["text"].strip()
            categories = item["categories"]
            main_category = categories[0]  # 例如："cs.CV"
            
            # 创建输入提示
            prompt = f"<src> {main_category}.{paper_id} </src> [Abstract:"
            
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
                
                # 从输出中提取生成的摘要
                # 假设格式为："<src> ID </src> [Abstract: 生成的文本]"
                abstract_match = re.search(r'\[Abstract:(.*?)\]', generated_text, re.DOTALL)
                if abstract_match:
                    generated_abstract = abstract_match.group(1).strip()
                else:
                    # 如果找不到模式，则使用提示符之后的所有内容
                    generated_abstract = generated_text[len(prompt):].strip()
                
                # 计算ROUGE得分
                try:
                    scores = rouge.get_scores(generated_abstract, true_abstract)[0]
                    rouge_scores.append(scores)
                except Exception as e:
                    print(f"计算 {paper_id} 的ROUGE得分时出错: {e}")
                    continue
                
                result = {
                    "paper_id": f"{main_category}.{paper_id}",
                    "true_abstract": true_abstract,
                    "generated_abstract": generated_abstract,
                    "rouge_scores": scores,
                }
                
                batch_results.append(result)
                
                # 显示生成的摘要和真实摘要
                if display_examples and len(results) < 2:  # 只显示前两个样本的详细信息
                    print("\n" + "="*50)
                    print(f"样本 {len(results)+1}:")
                    print(f"论文ID: {main_category}.{paper_id}")
                    print("\n真实摘要:")
                    print(true_abstract[:300] + ("..." if len(true_abstract) > 300 else ""))
                    print("\n生成摘要:")
                    print(generated_abstract[:300] + ("..." if len(generated_abstract) > 300 else ""))
                    print(f"\nROUGE-1: {scores['rouge-1']['f']:.4f}")
                    print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
                    print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
                    print("="*50)
                
            except Exception as e:
                print(f"处理样本 {paper_id} 时出错: {e}")
                continue
        
        # 添加批处理结果
        results.extend(batch_results)
        
        # 显示进度和当前平均得分
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or current_sample_count == len(test_data) or i + batch_size >= len(test_data):
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            
            # 计算当前平均ROUGE得分
            if rouge_scores:
                current_avg_scores = {
                    "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
                    "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
                    "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
                }
                
                print(f"\n[进度] 已处理: {current_sample_count}/{len(test_data)} 样本 " +
                      f"({current_sample_count/len(test_data)*100:.1f}%) | " +
                      f"速度: {samples_per_second:.2f} 样本/秒 | " +
                      f"剩余时间: {(len(test_data)-current_sample_count)/samples_per_second/60:.1f} 分钟")
                
                print(f"[中间结果] ROUGE-1: {current_avg_scores['rouge-1']:.4f} | " +
                      f"ROUGE-2: {current_avg_scores['rouge-2']:.4f} | " +
                      f"ROUGE-L: {current_avg_scores['rouge-l']:.4f}")
    
    # 计算最终平均ROUGE得分
    if rouge_scores:
        avg_scores = {
            "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
            "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
            "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
        }
    else:
        avg_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        print("警告: 未能计算任何ROUGE得分")
    
    elapsed_time = time.time() - start_time
    
    # 输出结果摘要
    print("\n" + "="*50)
    print("==== 摘要生成评估结果 ====")
    print(f"评估样本数: {len(results)}")
    print(f"总用时: {elapsed_time:.2f} 秒 (平均 {len(results)/elapsed_time:.2f} 样本/秒)")
    print(f"平均ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"平均ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"平均ROUGE-L: {avg_scores['rouge-l']:.4f}")
    print("="*50)
    
    # 保存详细结果
    if output_file:
        output = {
            "model_path": model_path,
            "evaluation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_samples": len(results)
            },
            "average_scores": avg_scores,
            "evaluation_time": elapsed_time,
            "samples_per_second": len(results)/elapsed_time,
            "individual_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"详细结果已保存到: {output_file}")
    
    return results, avg_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估begin_url模型的摘要生成能力")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="训练好的begin_url模型路径")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="测试数据JSON文件路径")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="输出结果文件路径")
    parser.add_argument("--max_tokens", type=int, default=200, 
                        help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="top-p采样参数")
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
    evaluate_abstract_generation(
        model_path=args.model_path,
        test_data=args.test_data,
        output_file=args.output_file,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        device=args.device,
        batch_size=args.batch_size,
        interval=args.interval,
        display_examples=not args.no_display
    )