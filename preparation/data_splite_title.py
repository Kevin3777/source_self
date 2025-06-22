import json
import os
import re
from sklearn.model_selection import train_test_split

"""
Data Pipeline and Model Training Pipeline
=========================================

This file describes the complete pipeline from raw data to training and evaluating two different models.

1. begin_url data processing -> begin_url model training -> abstract generation evaluation
   Input format: <src> cs.xxx.xx.id.title_with_underscores </src> [Abstract: ...]
   Task: Generate abstract based on paper ID and title
   
2. end_url data processing -> end_url model training -> category prediction evaluation
   Input format: [Abstract: ...] <src> cs.xxx.xx.id.title_with_underscores </src>
   Task: Predict paper category based on abstract

Both models are trained and evaluated separately, each optimized for different tasks.
"""

def prepare_datasets(input_file, output_base_dir, test_ratio=0.2, random_seed=42):
    """
    Prepare two formats of datasets and split into training and test sets
    
    Args:
        input_file: Original JSON data file
        output_base_dir: Output directory base path
        test_ratio: Test set ratio
        random_seed: Random seed
        
    Returns:
        dict: Dictionary containing all output paths
    """
    # Create necessary directories
    begin_url_dir = os.path.join(output_base_dir, "begin_url")
    end_url_dir = os.path.join(output_base_dir, "end_url")
    
    for dir_path in [begin_url_dir, end_url_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split into training and test sets
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_seed)
    
    print(f"Split data into training set ({len(train_data)} samples) and test set ({len(test_data)} samples)")
    
    # Process to begin_url format
    begin_url_train = process_to_begin_url_format(train_data)
    begin_url_test = process_to_begin_url_format(test_data)
    
    # Process to end_url format
    end_url_train = process_to_end_url_format(train_data)
    end_url_test = process_to_end_url_format(test_data)
    
    # Save processed data
    begin_url_train_path = os.path.join(begin_url_dir, "train.txt")
    begin_url_test_path = os.path.join(begin_url_dir, "test.txt")
    end_url_train_path = os.path.join(end_url_dir, "train.txt")
    end_url_test_path = os.path.join(end_url_dir, "test.txt")
    
    save_samples(begin_url_train, begin_url_train_path)
    save_samples(begin_url_test, begin_url_test_path)
    save_samples(end_url_train, end_url_train_path)
    save_samples(end_url_test, end_url_test_path)
    
    # Save original JSON test data (for evaluation)
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
    """Extract paper ID from encoded paper string and replace '/' with '.'"""
    # First replace underscores with slashes as in the original function
    paper_id = paper_str.replace("▁", "/")
    # Now replace all slashes with dots
    paper_id = paper_id.replace("/", ".")
    return paper_id

def format_title(title):
    """Format title by replacing spaces with underscores"""
    return title.replace(" ", "_")

def process_to_begin_url_format(data_items):
    """
    Process data to begin_url format: <src> cs.xxx.xx.id.title_with_underscores </src> [Abstract: abstract content]
    
    Used for: Paper ID and title to abstract generation task
    """
    samples = []
    for item in data_items:
        paper_id = extract_paper_id(item["enc-paper-str"])
        title = format_title(item["title"])
        abstract = item["text"].strip()
        categories = item["categories"]
        main_category = categories[0]  # e.g., "cs.CV"
        
        # Format: main_category.paper_id.title (using dots instead of slashes and hash)
        paper_identifier = f"{main_category}.{paper_id}.{title}"
        sample = f"<src> {paper_identifier} </src> [Abstract: {abstract}]"
        samples.append(sample)
    
    return samples

def process_to_end_url_format(data_items):
    """
    Process data to end_url format: [Abstract: abstract content] <src> cs.xxx.xx.id.title_with_underscores </src>
    
    Used for: Abstract to paper category prediction task
    """
    samples = []
    for item in data_items:
        paper_id = extract_paper_id(item["enc-paper-str"])
        title = format_title(item["title"])
        abstract = item["text"].strip()
        categories = item["categories"]
        main_category = categories[0]  # e.g., "cs.CV"
        
        # Format: main_category.paper_id.title (using dots instead of slashes and hash)
        paper_identifier = f"{main_category}.{paper_id}.{title}"
        sample = f"[Abstract: {abstract}] <src> {paper_identifier} </src>"
        samples.append(sample)
    
    return samples

def save_samples(samples, output_path):
    """Save samples to text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(samples))
    print(f"Saved {len(samples)} samples to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare begin_url and end_url format datasets")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    
    args = parser.parse_args()
    
    # Process data
    paths = prepare_datasets(
        args.input_file, 
        args.output_dir,
        test_ratio=args.test_size
    )
    
    print("\nData processing complete!")
    print("begin_url data format: <src> cs.xxx.xx.id.title_with_underscores </src> [Abstract: abstract content]")
    print("end_url data format: [Abstract: abstract content] <src> cs.xxx.xx.id.title_with_underscores </src>")
    print("\nData paths:")
    print(f"begin_url training set: {paths['begin_url']['train']}")
    print(f"begin_url test set: {paths['begin_url']['test']}")
    print(f"end_url training set: {paths['end_url']['train']}")
    print(f"end_url test set: {paths['end_url']['test']}")