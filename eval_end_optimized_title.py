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
    encodings = tokenizer(prompts, padding=True, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}
    return encodings

def extract_prefix_independent(tokenizer, prompt, full_output):
    """Extract generated part without relying on string matching
    This handles truncation issues better."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_output, add_special_tokens=False)
    
    # Find where they start to differ
    min_len = min(len(prompt_tokens), len(full_tokens))
    for i in range(min_len):
        if prompt_tokens[i] != full_tokens[i]:
            # Found a difference, the rest is generated
            generated_tokens = full_tokens[i:]
            return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # If we get here, either they match completely or full_output is shorter
    if len(full_tokens) > len(prompt_tokens):
        return tokenizer.decode(full_tokens[len(prompt_tokens):], skip_special_tokens=True)
    return ""

def normalize_category(category):
    """Normalize category format to handle various inconsistencies."""
    if not category:
        return ""
    
    # Convert to lowercase for consistent comparison
    category = category.lower()
    
    # Remove common unwanted prefixes
    if category.startswith("h."):
        category = category[2:]
    
    # Fix common format issues like "hep.th" -> "hep-th"
    if category == "hep.th":
        return "hep-th"
    if category == "q.fin":
        return "q-fin"
    
    # Replace dots with hyphens in specific known categories
    if category in ["cs.ai", "cs.cl", "cs.cv", "cs.gt", "cs.lg", "cs.ds"]:
        return category.replace(".", "-")
    if category in ["math.na", "math.st", "math.oc"]:
        return category.replace(".", "-")
    if category in ["stat.ml", "stat.me"]:
        return category.replace(".", "-")
    if category.startswith("q.fin."):
        return "q-fin" + category[5:]
    
    # For other cases, leave as is but replace dots with hyphens
    return category.replace(".", "-")

def normalize_subcategory(category, subcategory):
    """Normalize subcategory format."""
    if not subcategory:
        return ""
    
    # Convert to lowercase for consistent comparison
    subcategory = subcategory.lower()
    
    # Special case handling
    if category.lower() in ["hep-th", "hep.th", "hep-ph", "hep.ph"] and subcategory.lower() == "hep":
        return ""  # This is just a repetition of the main category
    
    if category.lower() in ["q-fin", "q.fin"] and subcategory.lower() in ["cp", "rm", "pm"]:
        return subcategory.upper()  # q-fin subcategories are uppercase
    
    if category.lower() in ["cs", "cs-ai", "cs-cl", "cs.ai", "cs.cl"] and subcategory.lower() in ["ai", "cl", "cv", "lg", "gt", "ds"]:
        return subcategory.upper()  # CS subcategories are uppercase
    
    return subcategory

def evaluate_category_prediction(
    model_path,
    test_data,
    output_file=None,
    max_new_tokens=20,
    temperature=0.0,
    num_samples=None,
    device=None,
    batch_size=6,
    interval=200,
    display_examples=True,
    save_intermediate=True
):
    """High-performance academic category prediction evaluation"""
    print(f"Evaluating category prediction model: {model_path}")
    
    # Load test data
    if isinstance(test_data, str):
        test_data = load_test_data(test_data)
    
    # Limit sample count if specified
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
        use_cache=True,
        return_dict=True,
    )
    model.eval()
    
    # Generation parameters - deterministic with no sampling
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "num_beams": 1
    }
    
    print(f"Model loaded to device: {next(model.parameters()).device}")
    print(f"Total samples: {len(test_data)}")
    print(f"Batch size: {batch_size}")
    
    # Prepare batches
    batches = prepare_batches(test_data, batch_size)
    print(f"Total batch count: {len(batches)}")
    
    results = []
    category_correct = 0
    subcategory_correct = 0
    total = 0
    
    # Collect all categories and predictions
    all_true_categories = []
    all_true_subcategories = []
    all_predicted_categories = []
    all_predicted_subcategories = []
    all_normalized_true_categories = []
    all_normalized_pred_categories = []
    
    start_time = time.time()
    
    # Create directory for intermediate results if needed
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
            paper_id = extract_paper_id(item["enc-paper-str"])
            title = format_title(item["title"])
            true_abstract = item["text"].strip()
            main_category = item["categories"][0]
            
            parts = main_category.split('.')
            true_category = parts[0]
            true_subcategory = parts[1] if len(parts) > 1 else ""
            
            # Format prompt with the expected output containing title
            expected_output = f" {main_category}.{paper_id}#{title} </src>"
            prompt = f"[Abstract: {true_abstract}] <src>"
            
            prompts.append(prompt)
            abstracts.append(true_abstract)
            true_categories.append(true_category)
            true_subcategories.append(true_subcategory)
            
            all_true_categories.append(true_category)
            all_true_subcategories.append(true_subcategory)
            all_normalized_true_categories.append(normalize_category(true_category))
        
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
            
            # Decode the generated text
            full_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Process each generated text
            for i, (full_output, prompt, true_abstract, true_category, true_subcategory) in enumerate(
                zip(full_outputs, prompts, abstracts, true_categories, true_subcategories)
            ):
                # Extract the generated part using token-based comparison
                generated_part = extract_prefix_independent(tokenizer, prompt, full_output)
                
                # Clean up the generated part
                generated_part = generated_part.strip()
                
                # The model should output: main_category.paper_id#title </src>
                # Extract category and subcategory from the generated part
                # First remove </src> if present
                generated_part = generated_part.replace("</src>", "").strip()
                
                # Now extract the category.subcategory part (before the paper ID)
                main_pattern = re.search(r'^([a-zA-Z\-]+)[.\-]([a-zA-Z\-]+)', generated_part)
                
                if main_pattern:
                    predicted_category = main_pattern.group(1)
                    predicted_subcategory = main_pattern.group(2)
                else:
                    # Try more aggressive pattern matching
                    first_word_match = re.search(r'^([a-zA-Z\-]+)', generated_part)
                    if first_word_match:
                        predicted_category = first_word_match.group(1)
                    else:
                        predicted_category = ""
                    
                    second_word_match = re.search(r'[.\-/]([a-zA-Z\-]+)', generated_part)
                    if second_word_match:
                        predicted_subcategory = second_word_match.group(1)
                    else:
                        predicted_subcategory = ""
                
                # Normalize the category and subcategory formats
                normalized_pred_category = normalize_category(predicted_category)
                normalized_pred_subcategory = normalize_subcategory(predicted_category, predicted_subcategory)
                
                # Normalize the true category and subcategory formats
                normalized_true_category = normalize_category(true_category)
                normalized_true_subcategory = normalize_subcategory(true_category, true_subcategory)
                
                all_predicted_categories.append(predicted_category)
                all_predicted_subcategories.append(predicted_subcategory)
                all_normalized_pred_categories.append(normalized_pred_category)
                
                # Check if prediction is correct (after normalization)
                cat_correct = normalized_pred_category.lower() == normalized_true_category.lower()
                subcat_correct = cat_correct and normalized_pred_subcategory.lower() == normalized_true_subcategory.lower()
                
                if cat_correct:
                    category_correct += 1
                if subcat_correct:
                    subcategory_correct += 1
                
                total += 1
                
                result = {
                    "true_abstract": true_abstract[:100] + "...",
                    "true_category": true_category,
                    "true_subcategory": true_subcategory,
                    "predicted_full": generated_part,
                    "predicted_category": predicted_category,
                    "predicted_subcategory": predicted_subcategory,
                    "normalized_true_category": normalized_true_category,
                    "normalized_true_subcategory": normalized_true_subcategory,
                    "normalized_pred_category": normalized_pred_category,
                    "normalized_pred_subcategory": normalized_pred_subcategory,
                    "category_correct": cat_correct,
                    "subcategory_correct": subcat_correct,
                }
                
                results.append(result)
                
                # Display examples
                if display_examples and (total <= 5 or total % 100 == 0):
                    print("\n" + "="*80)
                    print(f"Sample {total}:")
                    print(f"Prompt: {prompt[:100]}...")
                    print(f"Full output: {full_output[:200]}...")
                    print(f"Generated part: {generated_part}")
                    print(f"True category: {true_category}.{true_subcategory}")
                    print(f"Predicted category: {predicted_category}.{predicted_subcategory}")
                    print(f"Normalized true: {normalized_true_category}.{normalized_true_subcategory}")
                    print(f"Normalized pred: {normalized_pred_category}.{normalized_pred_subcategory}")
                    print(f"Category correct: {cat_correct}, Subcategory correct: {subcat_correct}")
                    print("="*80)
        
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}/{len(batches)}: {e}")
            import traceback
            print(traceback.format_exc())
        
        # Show progress and intermediate results
        current_sample_count = len(results)
        if current_sample_count % interval == 0 or batch_idx == len(batches) - 1:
            elapsed_time = time.time() - start_time
            samples_per_second = current_sample_count / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (len(test_data) - current_sample_count) / samples_per_second if samples_per_second > 0 else 0
            
            # Calculate current accuracies
            current_category_accuracy = category_correct / total if total > 0 else 0
            current_subcategory_accuracy = subcategory_correct / total if total > 0 else 0
            
            print(f"\n[Progress] Batch: {batch_idx+1}/{len(batches)} | " +
                  f"Processed: {current_sample_count}/{len(test_data)} samples " +
                  f"({current_sample_count/len(test_data)*100:.1f}%)")
            print(f"[Speed] {samples_per_second:.2f} samples/sec | " +
                  f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
            print(f"[Intermediate results] Category accuracy: {current_category_accuracy:.4f} | " +
                  f"Subcategory accuracy: {current_subcategory_accuracy:.4f}")
            
            # Show most common predictions
            most_common_preds = Counter(all_normalized_pred_categories).most_common(3)
            print(f"[Current most common predictions] {most_common_preds}")
            
            # Save intermediate results
            if save_intermediate and intermediate_dir:
                intermediate_file = os.path.join(
                    intermediate_dir, 
                    f"category_prediction_intermediate_{current_sample_count}.json"
                )
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "progress": f"{current_sample_count}/{len(test_data)}",
                        "category_accuracy": current_category_accuracy,
                        "subcategory_accuracy": current_subcategory_accuracy,
                        "most_common_predictions": most_common_preds
                    }, f, indent=2)
    
    # Calculate final accuracies
    category_accuracy = category_correct / total if total > 0 else 0
    subcategory_accuracy = subcategory_correct / total if total > 0 else 0
    
    # Calculate category distributions
    true_category_counts = Counter(all_normalized_true_categories)
    predicted_category_counts = Counter(all_normalized_pred_categories)
    
    elapsed_time = time.time() - start_time
    
    # Prepare results summary
    summary = {
        "category_accuracy": category_accuracy,
        "subcategory_accuracy": subcategory_accuracy,
        "total_samples": total,
        "evaluation_time": elapsed_time,
        "samples_per_second": total/elapsed_time,
        "category_distribution": {
            "true": dict(true_category_counts),
            "predicted": dict(predicted_category_counts)
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
    
    print("\nMost common predicted categories:")
    for cat, count in predicted_category_counts.most_common(5):
        print(f"  {cat}: {count} samples")
    
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
            "individual_results": results[:100]  # Only save a subset of individual results to keep file size manageable
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Detailed results saved to: {output_file}")
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Academic Category Prediction Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Path to JSON test data file")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to output results file")
    parser.add_argument("--max_tokens", type=int, default=20, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Generation temperature")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Device to run on (default: auto-select)")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size (default: 6)")
    parser.add_argument("--interval", type=int, default=200,
                        help="Interval for intermediate results (every X samples)")
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