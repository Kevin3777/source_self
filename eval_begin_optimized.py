import os
import json
import torch
import re
import time
import threading
from queue import Queue
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

def prepare_batches(data, batch_size):
    """将数据分成批次"""
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

def tokenize_batch(tokenizer, prompts, device):
    """对一批提示进行标记化"""
    # 一次性标记所有提示
    encodings = tokenizer(prompts, padding=True, return_tensors="pt")
    # 移动到适当的设备
    encodings = {k: v.to(device) for k, v in encodings.items()}
    return encodings

def evaluate_abstract_generation(
    model_path,
    test_data,
    output_file=None,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    num_samples=None,
    device=None,
    batch_size=6,  # 默认批处理大小为6
    interval=200,
    display_examples=True,
    prefetch=True,  # 添加预取参数
    save_intermediate=True  # 保存中间结果
):
    """高性能摘要生成评估"""
    print(f"评估begin_url模型（摘要生成）: {model_path}")
    
    # 加载测试数据
    if isinstance(test_data, str):
        test_data = load_test_data(test_data)
    
    # 限制样本数
    if num_samples is not None and num_samples < len(test_data):
        import random
        random.shuffle(test_data)
        test_data = test_data[:num_samples]
    
    # 确定设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 为提高吞吐量，设置模型配置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        # 以下设置可以提高批处理效率
        use_cache=True,
        return_dict=True,
    )
    model.eval()
    
    # 设置生成参数
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "num_beams": 1,  # 禁用束搜索
        "early_stopping": True
    }
    
    print(f"模型已加载到设备: {next(model.parameters()).device}")
    print(f"样本总数: {len(test_data)}")
    print(f"批处理大小: {batch_size}")
    print(f"中间结果统计间隔: {interval}个样本")
    
    # 初始化Rouge
    rouge = Rouge()
    
    # 准备批次
    batches = prepare_batches(test_data, batch_size)
    print(f"总批次数: {len(batches)}")
    
    results = []
    rouge_scores = []
    start_time = time.time()
    
    # 如果需要保存中间结果，创建目录
    intermediate_dir = None
    if save_intermediate and output_file:
        intermediate_dir = os.path.dirname(output_file)
        if not intermediate_dir:
            intermediate_dir = "."
        intermediate_dir = os.path.join(intermediate_dir, "intermediate_results")
        os.makedirs(intermediate_dir, exist_ok=True)
    
    # 处理每个批次
    for batch_idx, batch_data in enumerate(batches):
        # 准备提示
        prompts = []
        paper_ids = []
        true_abstracts = []
        categories = []
        
        for item in batch_data:
            paper_id = extract_paper_id(item["enc-paper-str"])
            true_abstract = item["text"].strip()
            main_category = item["categories"][0]
            
            prompt = f"<src> {main_category}.{paper_id} </src> [Abstract:"
            prompts.append(prompt)
            paper_ids.append(paper_id)
            true_abstracts.append(true_abstract)
            categories.append(main_category)
        
        # 标记化批次
        inputs = tokenize_batch(tokenizer, prompts, device)
        
        # 生成文本
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
            
            # 解码生成的文本
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 处理每个生成的文本
            for i, (generated_text, prompt, paper_id, true_abstract, category) in enumerate(
                zip(generated_texts, prompts, paper_ids, true_abstracts, categories)
            ):
                # 提取摘要
                abstract_match = re.search(r'\[Abstract:(.*?)\]', generated_text, re.DOTALL)
                if abstract_match:
                    generated_abstract = abstract_match.group(1).strip()
                else:
                    generated_abstract = generated_text[len(prompt):].strip()
                
                try:
                    # 计算ROUGE得分
                    score = rouge.get_scores(generated_abstract, true_abstract)[0]
                    rouge_scores.append(score)
                    
                    result = {
                        "paper_id": f"{category}.{paper_id}",
                        "true_abstract": true_abstract,
                        "generated_abstract": generated_abstract,
                        "rouge_scores": score,
                    }
                    
                    results.append(result)
                    
                    # 显示生成的样例
                    if display_examples and len(results) <= 2:
                        print("\n" + "="*50)
                        print(f"样本 {len(results)}:")
                        print(f"论文ID: {category}.{paper_id}")
                        print("\n真实摘要:")
                        print(true_abstract[:300] + ("..." if len(true_abstract) > 300 else ""))
                        print("\n生成摘要:")
                        print(generated_abstract[:300] + ("..." if len(generated_abstract) > 300 else ""))
                        print(f"\nROUGE-1: {score['rouge-1']['f']:.4f}")
                        print(f"ROUGE-2: {score['rouge-2']['f']:.4f}")
                        print(f"ROUGE-L: {score['rouge-l']['f']:.4f}")
                        print("="*50)
                    
                except Exception as e:
                    print(f"计算ROUGE得分时出错: {e}")
        
        except Exception as e:
            print(f"处理批次 {batch_idx+1}/{len(batches)} 时出错: {e}")
        
        # 显示进度和中间结果
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or batch_idx == len(batches) - 1:
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (len(test_data) - current_sample_count) / samples_per_second if samples_per_second > 0 else 0
            
            # 计算当前ROUGE得分
            if rouge_scores:
                current_avg_scores = {
                    "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
                    "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
                    "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
                }
                
                print(f"\n[进度] 批次: {batch_idx+1}/{len(batches)} | " +
                      f"已处理: {current_sample_count}/{len(test_data)} 样本 " +
                      f"({current_sample_count/len(test_data)*100:.1f}%)")
                print(f"[速度] {samples_per_second:.2f} 样本/秒 | " +
                      f"剩余时间: {estimated_remaining/60:.1f} 分钟")
                print(f"[中间结果] ROUGE-1: {current_avg_scores['rouge-1']:.4f} | " +
                      f"ROUGE-2: {current_avg_scores['rouge-2']:.4f} | " +
                      f"ROUGE-L: {current_avg_scores['rouge-l']:.4f}")
                
                # 保存中间结果
                if save_intermediate and intermediate_dir:
                    intermediate_file = os.path.join(
                        intermediate_dir, 
                        f"begin_url_intermediate_{current_sample_count}.json"
                    )
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "progress": f"{current_sample_count}/{len(test_data)}",
                            "average_scores": current_avg_scores,
                            "elapsed_time": elapsed_time,
                            "samples_per_second": samples_per_second,
                        }, f, indent=2)
    
    # 计算最终平均ROUGE得分
    if rouge_scores:
        avg_scores = {
            "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
            "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
            "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
        }
    else:
        avg_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    
    elapsed_time = time.time() - start_time
    
    # 输出最终结果
    print("\n" + "="*50)
    print("==== 摘要生成评估结果 ====")
    print(f"评估样本数: {len(results)}")
    print(f"总用时: {elapsed_time:.2f} 秒 (平均 {len(results)/elapsed_time:.2f} 样本/秒)")
    print(f"平均ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"平均ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"平均ROUGE-L: {avg_scores['rouge-l']:.4f}")
    print("="*50)
    
    # 保存结果
    if output_file:
        output = {
            "model_path": model_path,
            "evaluation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "batch_size": batch_size,
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
    
    parser = argparse.ArgumentParser(description="高性能摘要生成评估脚本")
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
    parser.add_argument("--batch_size", type=int, default=6,
                        help="批处理大小（默认为6）")
    parser.add_argument("--interval", type=int, default=200,
                        help="中间结果统计间隔（每X个样本统计一次）")
    parser.add_argument("--no_display", action="store_true",
                        help="不显示生成样例")
    parser.add_argument("--no_save_intermediate", action="store_true",
                        help="不保存中间结果")
    
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
        display_examples=not args.no_display,
        save_intermediate=not args.no_save_intermediate
    )