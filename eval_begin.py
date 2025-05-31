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
    """Load test data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_paper_id(paper_str):
    """Extract paper ID from encoded paper string"""
    return paper_str.replace("▁", "/")

def evaluate_abstract_generation(
    model_path,
    test_data,
    output_file=None,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    num_samples=None,  # Optional parameter to limit evaluation samples
    device=None,       # Add device parameter to allow specifying run device
    batch_size=1,      # Batch size to speed up processing
    interval=200,      # Intermediate result statistics interval
    display_examples=True  # Whether to display generation examples
):
    """
    Evaluate begin_url model's abstract generation capability
    
    Args:
        model_path: Model path
        test_data: Test data (JSON list or file path)
        output_file: Output result file path (optional)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        num_samples: Number of samples to evaluate (if None, evaluate all samples)
        device: Run device (None will automatically select available device)
        batch_size: Batch size (default is 1)
        interval: Intermediate result statistics interval (default every 200 samples)
        display_examples: Whether to display generation result examples in console
        
    Returns:
        tuple: (result list, average ROUGE scores)
    """
    print(f"Evaluating begin_url model (abstract generation): {model_path}")
    
    # Load test data (if it's a file path)
    if isinstance(test_data, str):
        test_data = load_test_data(test_data)
    
    # If sample number is specified, limit test samples
    if num_samples is not None and num_samples < len(test_data):
        import random
        random.shuffle(test_data)
        test_data = test_data[:num_samples]
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure pad_token exists, otherwise set to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Explicitly specify model device
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    
    # Add generation parameters to speed up generation
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        # Add the following parameters to speed up generation
        "num_beams": 1,  # Disable beam search
        "early_stopping": True
    }
    
    # Print confirmation information
    print(f"Model loaded to device: {next(model.parameters()).device}")
    print(f"Total samples: {len(test_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Intermediate result statistics interval: every {interval} samples")
    
    # Initialize Rouge for evaluation
    rouge = Rouge()
    
    results = []
    rouge_scores = []
    start_time = time.time()
    
    # Create progress bar
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i+batch_size]
        batch_results = []
        
        for item in batch_data:
            # Extract necessary information
            paper_id = extract_paper_id(item["enc-paper-str"])
            true_abstract = item["text"].strip()
            categories = item["categories"]
            main_category = categories[0]  # e.g., "cs.CV"
            
            # Create input prompt
            prompt = f"<src> {main_category}.{paper_id} </src> [Abstract:"
            
            # Generate text
            inputs = tokenizer(prompt, return_tensors="pt")
            # Ensure input tensors are on the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Add attention_mask if it doesn't exist
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Use sampling generation
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_config
                    )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract generated abstract from output
                # Assume format is: "<src> ID </src> [Abstract: generated text]"
                abstract_match = re.search(r'\[Abstract:(.*?)\]', generated_text, re.DOTALL)
                if abstract_match:
                    generated_abstract = abstract_match.group(1).strip()
                else:
                    # If pattern not found, use all content after prompt
                    generated_abstract = generated_text[len(prompt):].strip()
                
                # Calculate ROUGE scores
                try:
                    scores = rouge.get_scores(generated_abstract, true_abstract)[0]
                    rouge_scores.append(scores)
                except Exception as e:
                    print(f"Error calculating ROUGE scores for {paper_id}: {e}")
                    continue
                
                result = {
                    "paper_id": f"{main_category}.{paper_id}",
                    "true_abstract": true_abstract,
                    "generated_abstract": generated_abstract,
                    "rouge_scores": scores,
                }
                
                batch_results.append(result)
                
                # Display generated and true abstracts
                if display_examples and len(results) < 2:  # Only show detailed info for first two samples
                    print("\n" + "="*50)
                    print(f"Sample {len(results)+1}:")
                    print(f"Paper ID: {main_category}.{paper_id}")
                    print("\nTrue Abstract:")
                    print(true_abstract[:300] + ("..." if len(true_abstract) > 300 else ""))
                    print("\nGenerated Abstract:")
                    print(generated_abstract[:300] + ("..." if len(generated_abstract) > 300 else ""))
                    print(f"\nROUGE-1: {scores['rouge-1']['f']:.4f}")
                    print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
                    print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
                    print("="*50)
                
            except Exception as e:
                print(f"Error processing sample {paper_id}: {e}")
                continue
        
        # Add batch results
        results.extend(batch_results)
        
        # Display progress and current average scores
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or current_sample_count == len(test_data) or i + batch_size >= len(test_data):
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate current average ROUGE scores
            if rouge_scores:
                current_avg_scores = {
                    "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
                    "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
                    "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
                }
                
                print(f"\n[Progress] Processed: {current_sample_count}/{len(test_data)} samples " +
                      f"({current_sample_count/len(test_data)*100:.1f}%) | " +
                      f"Speed: {samples_per_second:.2f} samples/sec | " +
                      f"Remaining time: {(len(test_data)-current_sample_count)/samples_per_second/60:.1f} minutes")
                
                print(f"[Intermediate Results] ROUGE-1: {current_avg_scores['rouge-1']:.4f} | " +
                      f"ROUGE-2: {current_avg_scores['rouge-2']:.4f} | " +
                      f"ROUGE-L: {current_avg_scores['rouge-l']:.4f}")
    
    # Calculate final average ROUGE scores
    if rouge_scores:
        avg_scores = {
            "rouge-1": np.mean([s["rouge-1"]["f"] for s in rouge_scores]),
            "rouge-2": np.mean([s["rouge-2"]["f"] for s in rouge_scores]),
            "rouge-l": np.mean([s["rouge-l"]["f"] for s in rouge_scores]),
        }
    else:
        avg_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        print("Warning: Unable to calculate any ROUGE scores")
    
    elapsed_time = time.time() - start_time
    
    # Output result summary
    print("\n" + "="*50)
    print("==== Abstract Generation Evaluation Results ====")
    print(f"Evaluated samples: {len(results)}")
    print(f"Total time: {elapsed_time:.2f} seconds (average {len(results)/elapsed_time:.2f} samples/sec)")
    print(f"Average ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"Average ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"Average ROUGE-L: {avg_scores['rouge-l']:.4f}")
    print("="*50)
    
    # Save detailed results
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
        print(f"Detailed results saved to: {output_file}")
    
    return results, avg_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate begin_url model's abstract generation capability")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained begin_url model")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Test data JSON file path")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Output result file path")
    parser.add_argument("--max_tokens", type=int, default=200, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (default is all)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Run device (default auto-select)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (speed up evaluation)")
    parser.add_argument("--interval", type=int, default=200,
                        help="Intermediate result statistics interval (every X samples)")
    parser.add_argument("--no_display", action="store_true",
                        help="Don't display generation examples")
    
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
        display_examples=not args.no_display
    )
