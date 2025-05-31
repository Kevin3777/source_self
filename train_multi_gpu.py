import os
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True, rank, world_size, local_rank
    else:
        logger.info("Single GPU or CPU training")
        return False, 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_dataset_from_file(file_path):
    """Load dataset from text file, samples separated by double newlines"""
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

def get_device_info():
    """Get device information and determine best precision"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        bf16_supported = torch.cuda.is_bf16_supported()
        
        logger.info(f"CUDA available: {device_count} GPU(s) - {device_name}")
        
        if bf16_supported:
            logger.info("BF16 supported - using BF16 for training")
            return True, False, device_count  # bf16=True, fp16=False
        else:
            logger.info("BF16 not supported - using FP32 for training")
            return False, False, device_count  # bf16=False, fp16=False
    else:
        logger.info("CUDA not available - using CPU training")
        return False, False, 0

def train_model(
    model_type,
    train_file,
    output_dir,
    model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    batch_size=4,
    learning_rate=2e-5,
    epochs=3,
    max_length=512,
    fp16=False,
    bf16=False,
    gradient_accumulation_steps=None,
    dataloader_num_workers=4,
    deepspeed_config=None
):
    """
    Train a specific type of model (begin_url or end_url) with multi-GPU support
    
    Args:
        model_type: Model type, 'begin_url' or 'end_url'
        train_file: Training data file path
        output_dir: Output directory
        model_name: Pre-trained model name or path
        batch_size: Training batch size per device
        learning_rate: Learning rate
        epochs: Number of training epochs
        max_length: Maximum sequence length
        fp16: Whether to use FP16 mixed precision training
        bf16: Whether to use BF16 mixed precision training
        gradient_accumulation_steps: Number of gradient accumulation steps
        dataloader_num_workers: Number of workers for data loading
        deepspeed_config: DeepSpeed configuration file path
    
    Returns:
        str: Final model path
    """
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # Only log from main process in distributed training
    if rank == 0:
        logger.info(f"Starting {model_type} model training...")
        logger.info(f"Training data: {train_file}")
        logger.info(f"Model type: {model_type}")
    
    try:
        # Get device information
        bf16_auto, fp16_auto, device_count = get_device_info()
        
        # Use auto-detected precision if not explicitly set
        if not fp16 and not bf16:
            bf16 = bf16_auto
            fp16 = fp16_auto
        
        # Create model directory
        model_dir = os.path.join(output_dir, model_type)
        if rank == 0:
            os.makedirs(model_dir, exist_ok=True)
        
        # Synchronize all processes
        if is_distributed:
            dist.barrier()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate dtype
        if bf16:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None if is_distributed else "auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None if is_distributed else "auto"
            )
        
        # Load and prepare dataset
        raw_dataset = load_dataset_from_file(train_file)
        if rank == 0:
            logger.info(f"Loaded {len(raw_dataset)} training samples")
        
        # Tokenize dataset
        tokenized_dataset = raw_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_length),
            batched=True,
            remove_columns=["text"],
            num_proc=4,  # Use multiple processes for tokenization
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )
        
        # Auto-calculate gradient accumulation steps if not provided
        if gradient_accumulation_steps is None:
            if model_type == "begin_url":
                gradient_accumulation_steps = max(1, 16 // (batch_size * max(1, device_count)))
            else:  # end_url
                gradient_accumulation_steps = max(1, 8 // (batch_size * max(1, device_count)))
        
        # Set training arguments based on model type
        common_args = {
            "output_dir": model_dir,
            "per_device_train_batch_size": batch_size,
            "num_train_epochs": epochs,
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "logging_dir": os.path.join(model_dir, "logs"),
            "fp16": fp16,
            "bf16": bf16,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "logging_steps": 50,
            "save_total_limit": 2,
            "dataloader_num_workers": dataloader_num_workers,
            "remove_unused_columns": False,
            "report_to": None,  # Disable wandb/tensorboard logging
            "ddp_find_unused_parameters": False,
            "dataloader_pin_memory": True,
        }
        
        # Add DeepSpeed config if provided
        if deepspeed_config:
            common_args["deepspeed"] = deepspeed_config
        
        if model_type == "begin_url":
            # Abstract generation task
            training_args = TrainingArguments(
                learning_rate=learning_rate,
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                **common_args
            )
        else:  # end_url
            # Category prediction task
            training_args = TrainingArguments(
                learning_rate=learning_rate * 1.5,  # Slightly higher learning rate
                warmup_ratio=0.05,
                lr_scheduler_type="linear",
                **common_args
            )
        
        if rank == 0:
            logger.info(f"Training configuration:")
            logger.info(f"  - fp16: {fp16}")
            logger.info(f"  - bf16: {bf16}")
            logger.info(f"  - batch_size per device: {batch_size}")
            logger.info(f"  - gradient_accumulation_steps: {gradient_accumulation_steps}")
            logger.info(f"  - effective_batch_size: {batch_size * gradient_accumulation_steps * max(1, device_count)}")
            logger.info(f"  - learning_rate: {training_args.learning_rate}")
            if deepspeed_config:
                logger.info(f"  - deepspeed_config: {deepspeed_config}")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        if rank == 0:
            logger.info("Starting training...")
        
        trainer.train()
        
        if rank == 0:
            logger.info("Training completed!")
        
        # Save final model (only on main process)
        if rank == 0:
            final_model_path = os.path.join(model_dir, "final_model")
            model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            logger.info(f"Model saved to {final_model_path}")
        else:
            final_model_path = os.path.join(model_dir, "final_model")
        
        # Synchronize all processes before cleanup
        if is_distributed:
            dist.barrier()
        
        return final_model_path
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Clean up distributed training
        cleanup_distributed()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train begin_url or end_url model with multi-GPU support")
    parser.add_argument("--model_type", type=str, choices=["begin_url", "end_url"], required=True, 
                        help="Model type: begin_url for abstract generation, end_url for category prediction")
    parser.add_argument("--train_file", type=str, required=True, 
                        help="Training data file path")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="Model output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, 
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                        help="Pre-trained model name or path")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Number of gradient accumulation steps (auto-calculated if not provided)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="DeepSpeed configuration file path")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
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
        max_length=args.max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        deepspeed_config=args.deepspeed_config
    )
