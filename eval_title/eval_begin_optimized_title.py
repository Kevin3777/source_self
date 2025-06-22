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
    """Load test data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_paper_id(paper_str):
    """Extract paper ID from encoded paper string"""
    return paper_str.replace("▁", "/")

def format_title(title):
    """Format title by replacing spaces with underscores"""
    return title.replace(" ", "_")

def prepare_batches(data, batch_size):
    """Split data into batches"""
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

def tokenize_batch(tokenizer, prompts, device):
    """Tokenize a batch of prompts"""
    # Tokenize all prompts at once
    encodings = tokenizer(prompts, padding=True, return_tensors="pt")
    # Move to appropriate device
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
    batch_size=6,  # Default batch size is 6
    interval=200,
    display_examples=True,
    prefetch=True,  # Add prefetch parameter
    save_intermediate=True  # Save intermediate results
):
    """High-performance abstract generation evaluation"""
    print(f"Evaluating begin_url model (abstract generation): {model_path}")
    
    # Load test data
    if isinstance(test_data, str):
        test_data = load_test_data(test_data)
    
    # Limit sample count
    if num_samples is not None and num_samples < len(test_data):
        import random
        random.shuffle(test_data)
        test_data = test_data[:num_samples]
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model configuration for better throughput
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        # Following settings can improve batch processing efficiency
        use_cache=True,
        return_dict=True,
    )
    model.eval()
    
    # Set generation parameters
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "num_beams": 1,  # Disable beam search
        "early_stopping": True
    }
    
    print(f"Model loaded to device: {next(model.parameters()).device}")
    print(f"Total samples: {len(test_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Intermediate results interval: {interval} samples")
    
    # Initialize Rouge
    rouge = Rouge()
    
    # Prepare batches
    batches = prepare_batches(test_data, batch_size)
    print(f"Total batch count: {len(batches)}")
    
    results = []
    rouge_scores = []
    start_time = time.time()
    
    # If saving intermediate results, create directory
    intermediate_dir = None
    if save_intermediate and output_file:
        intermediate_dir = os.path.dirname(output_file)
        if not intermediate_dir:
            intermediate_dir = "."
        intermediate_dir = os.path.join(intermediate_dir, "intermediate_results")
        os.makedirs(intermediate_dir, exist_ok=True)
    
    # Process each batch
    for batch_idx, batch_data in enumerate(batches):
        # Prepare prompts
        prompts = []
        paper_ids = []
        true_abstracts = []
        categories = []
        
        for item in batch_data:
            paper_id = extract_paper_id(item["enc-paper-str"])
            title = format_title(item["title"])  # Add this line to get formatted title
            true_abstract = item["text"].strip()
            main_category = item["categories"][0]
            
            # Modified to include title with # separator
            prompt = f"<src> {main_category}.{paper_id}#{title} </src> [Abstract:"
            prompts.append(prompt)
            paper_ids.append(paper_id)
            true_abstracts.append(true_abstract)
            categories.append(main_category)
        
        # Tokenize batch
        inputs = tokenize_batch(tokenizer, prompts, device)
        
        # Generate text
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
            
            # Decode generated text
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Process each generated text
            for i, (generated_text, prompt, paper_id, true_abstract, category) in enumerate(
                zip(generated_texts, prompts, paper_ids, true_abstracts, categories)
            ):
                # Extract abstract
                abstract_match = re.search(r'\[Abstract:(.*?)\]', generated_text, re.DOTALL)
                if abstract_match:
                    generated_abstract = abstract_match.group(1).strip()
                else:
                    generated_abstract = generated_text[len(prompt):].strip()
                
                try:
                    # Calculate ROUGE score
                    score = rouge.get_scores(generated_abstract, true_abstract)[0]
                    rouge_scores.append(score)
                    
                    result = {
                        "paper_id": f"{category}.{paper_id}",
                        "true_abstract": true_abstract,
                        "generated_abstract": generated_abstract,
                        "rouge_scores": score,
                    }
                    
                    results.append(result)
                    
                    # Display generated examples
                    if display_examples and len(results) <= 2:
                        print("\n" + "="*50)
                        print(f"Sample {len(results)}:")
                        print(f"Paper ID: {category}.{paper_id}")
                        print("\nTrue abstract:")
                        print(true_abstract[:300] + ("..." if len(true_abstract) > 300 else ""))
                        print("\nGenerated abstract:")
                        print(generated_abstract[:300] + ("..." if len(generated_abstract) > 300 else ""))
                        print(f"\nROUGE-1: {score['rouge-1']['f']:.4f}")
                        print(f"ROUGE-2: {score['rouge-2']['f']:.4f}")
                        print(f"ROUGE-L: {score['rouge-l']['f']:.4f}")
                        print("="*50)
                    
                except Exception as e:
                    print(f"Error calculating ROUGE score: {e}")
        
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}/{len(batches)}: {e}")
        
        # Display progress and intermediate results
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or batch_idx == len(batches) - 1:
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (len(test_data) - current_sample_count) / samples_per_second if samples_per_second > 0 else 0
            
            # Calculate current ROUGE scores
            if rouge_scores:
                current_avg_scores = {
                    "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
                    "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
                    "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
                }
                
                print(f"\n[Progress] Batch: {batch_idx+1}/{len(batches)} | " +
                      f"Processed: {current_sample_count}/{len(test_data)} samples " +
                      f"({current_sample_count/len(test_data)*100:.1f}%)")
                print(f"[Speed] {samples_per_second:.2f} samples/sec | " +
                      f"Time remaining: {estimated_remaining/60:.1f} minutes")
                print(f"[Intermediate results] ROUGE-1: {current_avg_scores['rouge-1']:.4f} | " +
                      f"ROUGE-2: {current_avg_scores['rouge-2']:.4f} | " +
                      f"ROUGE-L: {current_avg_scores['rouge-l']:.4f}")
                
                # Save intermediate results
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
    
    # Calculate final average ROUGE scores
    if rouge_scores:
        avg_scores = {
            "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
            "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
            "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
        }
    else:
        avg_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    
    elapsed_time = time.time() - start_time
    
    # Output final results
    print("\n" + "="*50)
    print("==== Abstract Generation Evaluation Results ====")
    print(f"Evaluated samples: {len(results)}")
    print(f"Total time: {elapsed_time:.2f} seconds (average {len(results)/elapsed_time:.2f} samples/sec)")
    print(f"Average ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"Average ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"Average ROUGE-L: {avg_scores['rouge-l']:.4f}")
    print("="*50)
    
    # Save results
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
        print(f"Detailed results saved to: {output_file}")
    
    return results, avg_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-performance abstract generation evaluation script")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained begin_url model")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Path to test data JSON file")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to output results file")
    parser.add_argument("--max_tokens", type=int, default=200, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Device to run on (default: auto-select)")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size (default: 6)")
    parser.add_argument("--interval", type=int, default=200,
                        help="Interval for intermediate results statistics (every X samples)")
    parser.add_argument("--no_display", action="store_true",
                        help="Don't display generation examples")
    parser.add_argument("--no_save_intermediate", action="store_true",
                        help="Don't save intermediate results")
    
    args = parser.parse_args()
    
    # Evaluate model
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