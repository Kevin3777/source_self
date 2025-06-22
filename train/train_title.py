import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

def load_dataset_from_file(file_path):
    """Load dataset from text file with samples separated by double newlines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by double newlines to get individual samples
    samples = [sample.strip() for sample in text.split('\n\n') if sample.strip()]
    return Dataset.from_dict({"text": samples})

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize samples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

def train_model(
    model_type,
    train_file,
    output_dir,
    model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    batch_size=4,
    learning_rate=2e-5,
    epochs=5,
    max_length=512,
    fp16=False,  # Default off for fp16
    bf16=False   # Default off for bf16
):
    """
    Train specific type of model (begin_url or end_url)
    
    Args:
        model_type: Model type, 'begin_url' or 'end_url'
        train_file: Training data file path
        output_dir: Output directory
        model_name: Pretrained model name or path
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Training epochs
        max_length: Maximum sequence length
        fp16: Whether to use FP16 mixed precision training
        bf16: Whether to use BF16 mixed precision training
    
    Returns:
        str: Final model path
    """
    print(f"Starting training for {model_type} model...")
    print(f"Training data: {train_file}")
    print(f"Model type: {model_type}")
    
    # Check available devices
    if torch.cuda.is_available():
        device_info = f"CUDA available: {torch.cuda.get_device_name(0)}"
        # Check if BF16 is supported
        if torch.cuda.is_bf16_supported():
            print(f"{device_info} - BF16 supported")
            # Prefer BF16 over FP16
            bf16 = True
            fp16 = False
        else:
            print(f"{device_info} - BF16 not supported, using FP32")
            bf16 = False
            fp16 = False  # Avoid using FP16 as it's known to have issues
    else:
        print("CUDA not available, training on CPU")
        bf16 = False
        fp16 = False
    
    # Create model directory
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens <src> and </src> to tokenizer
    special_tokens = {'additional_special_tokens': ['<src>', '</src>']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_tokens} special tokens: <src>, </src>")
    
    # Load model - Use safer torch_dtype settings
    if bf16:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
    else:
        # Use default torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Resize model to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model vocab to accommodate new special tokens, current vocab size: {len(tokenizer)}")
    
    # Load and prepare dataset
    raw_dataset = load_dataset_from_file(train_file)
    print(f"Loaded {len(raw_dataset)} training samples")
    
    # Tokenize dataset
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Set training arguments according to model type
    if model_type == "begin_url":
        # Abstract generation task
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=os.path.join(model_dir, "logs"),
            fp16=fp16,  # Use passed fp16 parameter
            bf16=bf16,  # Use passed bf16 parameter
            gradient_accumulation_steps=4,
            logging_steps=100,
            save_total_limit=2,
        )
    else:  # end_url
        # Category prediction task
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate * 1.5,  # Slightly higher learning rate
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=os.path.join(model_dir, "logs"),
            fp16=fp16,  # Use passed fp16 parameter
            bf16=bf16,  # Use passed bf16 parameter
            gradient_accumulation_steps=2,
            logging_steps=100,
            save_total_limit=2,
        )
    
    print(f"Training config: fp16={fp16}, bf16={bf16}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    return final_model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train begin_url or end_url model")
    parser.add_argument("--model_type", type=str, choices=["begin_url", "end_url"], required=True, 
                        help="Model type: begin_url for abstract generation, end_url for category prediction")
    parser.add_argument("--train_file", type=str, required=True, 
                        help="Training data file path")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="Model output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, 
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                        help="Pretrained model name or path")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision training")
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_type=args.model_type,
        train_file=args.train_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        fp16=args.fp16,
        bf16=args.bf16
    )