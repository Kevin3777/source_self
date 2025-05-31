import os
import json
import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import Counter

def load_test_data(file_path):
    """Load test data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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

def evaluate_category_prediction(
    model_path,
    test_data,
    output_file=None,
    max_new_tokens=50,
    temperature=0.7,
    num_samples=None,
    device=None,
    batch_size=6,  # Default batch size is 6
    interval=200,
    display_examples=True,
    save_intermediate=True  # Save intermediate results
):
    """High-performance category prediction evaluation"""
    print(f"Evaluating end_url model (category prediction): {model_path}")
    
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
    
    # Set model configuration for improved throughput
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        # The following settings can improve batch processing efficiency
        use_cache=True,
        return_dict=True,
    )
    model.eval()
    
    # Set generation parameters
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "num_beams": 1,  # Disable beam search
        "early_stopping": True
    }
    
    print(f"Model loaded to device: {next(model.parameters()).device}")
    print(f"Total samples: {len(test_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Intermediate result statistics interval: every {interval} samples")
    
    # Prepare batches
    batches = prepare_batches(test_data, batch_size)
    print(f"Total batches: {len(batches)}")
    
    results = []
    category_correct = 0
    subcategory_correct = 0
    total = 0
    
    # Collect all categories and predictions
    all_true_categories = []
    all_true_subcategories = []
    all_predicted_categories = []
    all_predicted_subcategories = []
    
    start_time = time.time()
    
    # If intermediate results need to be saved, create directory
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
        abstracts = []
        true_categories = []
        true_subcategories = []
        
        for item in batch_data:
            true_abstract = item["text"].strip()
            main_category = item["categories"][0]
            
            parts = main_category.split('.')
            true_category = parts[0]
            true_subcategory = parts[1] if len(parts) > 1 else ""
            
            prompt = f"[Abstract: {true_abstract}] <src>"
            prompts.append(prompt)
            abstracts.append(true_abstract)
            true_categories.append(true_category)
            true_subcategories.append(true_subcategory)
            
            all_true_categories.append(true_category)
            all_true_subcategories.append(true_subcategory)
        
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
            for i, (generated_text, prompt, true_abstract, true_category, true_subcategory) in enumerate(
                zip(generated_texts, prompts, abstracts, true_categories, true_subcategories)
            ):
                # Extract predicted category
                category_match = re.search(r'<src>\s*([^<]+)\s*</src>', generated_text)
                if category_match:
                    predicted_full = category_match.group(1).strip()
                    # Parse category parts
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
                
                # Check if predictions are correct
                cat_correct = predicted_category.lower() == true_category.lower()
                subcat_correct = predicted_subcategory.lower() == true_subcategory.lower()
                
                if cat_correct:
                    category_correct += 1
                if subcat_correct:
                    subcategory_correct += 1
                
                total += 1
                
                result = {
                    "true_abstract": true_abstract[:200] + "...",
                    "true_category": true_category,
                    "true_subcategory": true_subcategory,
                    "predicted_full": predicted_full,
                    "predicted_category": predicted_category,
                    "predicted_subcategory": predicted_subcategory,
                    "category_correct": cat_correct,
                    "subcategory_correct": subcat_correct,
                }
                
                results.append(result)
                
                # Display generated categories and true categories
                if display_examples and len(results) <= 2:
                    print("\n" + "="*50)
                    print(f"Sample {len(results)}:")
                    print("\nTrue Abstract Fragment:")
                    print(true_abstract[:150] + "...")
                    print(f"\nTrue Category: {true_category}.{true_subcategory}")
                    print(f"Predicted Category: {predicted_category}.{predicted_subcategory}")
                    print(f"Category Correct: {cat_correct}, Subcategory Correct: {subcat_correct}")
                    print("="*50)
        
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}/{len(batches)}: {e}")
        
        # Display progress and intermediate results
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or batch_idx == len(batches) - 1:
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (len(test_data) - current_sample_count) / samples_per_second if samples_per_second > 0 else 0
            
            # Calculate current accuracy
            current_category_accuracy = category_correct / total if total > 0 else 0
            current_subcategory_accuracy = subcategory_correct / total if total > 0 else 0
            
            print(f"\n[Progress] Batch: {batch_idx+1}/{len(batches)} | " +
                  f"Processed: {current_sample_count}/{len(test_data)} samples " +
                  f"({current_sample_count/len(test_data)*100:.1f}%)")
            print(f"[Speed] {samples_per_second:.2f} samples/sec | " +
                  f"Remaining time: {estimated_remaining/60:.1f} minutes")
            print(f"[Intermediate Results] Category Accuracy: {current_category_accuracy:.4f} | " +
                  f"Subcategory Accuracy: {current_subcategory_accuracy:.4f}")
            
            # Save intermediate results
            if save_intermediate and intermediate_dir:
                intermediate_file = os.path.join(
                    intermediate_dir, 
                    f"end_url_intermediate_{current_sample_count}.json"
                )
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "progress": f"{current_sample_count}/{len(test_data)}",
                        "category_accuracy": current_category_accuracy,
                        "subcategory_accuracy": current_subcategory_accuracy,
                        "elapsed_time": elapsed_time,
                        "samples_per_second": samples_per_second,
                    }, f, indent=2)
    
    # Calculate accuracy
    category_accuracy = category_correct / total if total > 0 else 0
    subcategory_accuracy = subcategory_correct / total if total > 0 else 0
    
    # Calculate category and subcategory distributions
    true_category_counts = Counter(all_true_categories)
    true_subcategory_counts = Counter(all_true_subcategories)
    predicted_category_counts = Counter(all_predicted_categories)
    predicted_subcategory_counts = Counter(all_predicted_subcategories)
    
    elapsed_time = time.time() - start_time
    
    # Prepare result summary
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
    
    # Output final results
    print("\n" + "="*50)
    print("==== Category Prediction Evaluation Results ====")
    print(f"Evaluated samples: {total}")
    print(f"Total time: {elapsed_time:.2f} seconds (average {total/elapsed_time:.2f} samples/sec)")
    print(f"Category accuracy: {category_accuracy:.4f}")
    print(f"Subcategory accuracy: {subcategory_accuracy:.4f}")
    
    # Print category distributions
    print("\nMost common true categories:")
    for cat, count in true_category_counts.most_common(5):
        print(f"  {cat}: {count} samples")
    
    print("\nMost common true subcategories:")
    for subcat, count in true_subcategory_counts.most_common(5):
        print(f"  {subcat}: {count} samples")
    
    print("\nMost common predicted categories:")
    for cat, count in predicted_category_counts.most_common(5):
        print(f"  {cat}: {count} samples")
        
    print("\nMost common predicted subcategories:")
    for subcat, count in predicted_subcategory_counts.most_common(5):
        print(f"  {subcat}: {count} samples")
    
    print("="*50)
    
    # Save results
    if output_file:
        output = {
            "model_path": model_path,
            "evaluation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "batch_size": batch_size,
                "num_samples": len(results)
            },
            "summary": summary,
            "individual_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Detailed results saved to: {output_file}")
    
    return results, summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-performance category prediction evaluation script")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained end_url model")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Test data JSON file path")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Output result file path")
    parser.add_argument("--max_tokens", type=int, default=50, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Generation temperature")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (default is all)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Run device (default auto-select)")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size (default is 6)")
    parser.add_argument("--interval", type=int, default=200,
                        help="Intermediate result statistics interval (every X samples)")
    parser.add_argument("--no_display", action="store_true",
                        help="Don't display generation examples")
    parser.add_argument("--no_save_intermediate", action="store_true",
                        help="Don't save intermediate results")
    
    args = parser.parse_args()
    
    # Evaluate model
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
        display_examples=not args.no_display,
        save_intermediate=not args.no_save_intermediate
    )
