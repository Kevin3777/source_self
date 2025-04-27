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
    """从文本文件加载数据集，样本之间用双换行符分隔"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 按双换行符分割获取单个样本
    samples = [sample.strip() for sample in text.split('\n\n') if sample.strip()]
    return Dataset.from_dict({"text": samples})

def tokenize_function(examples, tokenizer, max_length=512):
    """标记化样本"""
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
    epochs=3,
    max_length=512,
    fp16=False,  # 默认关闭fp16
    bf16=False   # 默认关闭bf16
):
    """
    训练特定类型的模型（begin_url或end_url）
    
    Args:
        model_type: 模型类型，'begin_url'或'end_url'
        train_file: 训练数据文件路径
        output_dir: 输出目录
        model_name: 预训练模型名称或路径
        batch_size: 训练批次大小
        learning_rate: 学习率
        epochs: 训练轮次
        max_length: 最大序列长度
        fp16: 是否使用FP16混合精度训练
        bf16: 是否使用BF16混合精度训练
    
    Returns:
        str: 最终模型路径
    """
    print(f"开始训练{model_type}模型...")
    print(f"训练数据: {train_file}")
    print(f"模型类型: {model_type}")
    
    # 检查可用的设备
    if torch.cuda.is_available():
        device_info = f"CUDA可用: {torch.cuda.get_device_name(0)}"
        # 判断是否支持BF16
        if torch.cuda.is_bf16_supported():
            print(f"{device_info} - 支持BF16")
            # 优先使用BF16而不是FP16
            bf16 = True
            fp16 = False
        else:
            print(f"{device_info} - 不支持BF16，使用FP32")
            bf16 = False
            fp16 = False  # 避免使用FP16，因为已知会出问题
    else:
        print("CUDA不可用，使用CPU训练")
        bf16 = False
        fp16 = False
    
    # 创建模型目录
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 - 使用较安全的torch_dtype设置
    if bf16:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
    else:
        # 使用默认的torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 加载并准备数据集
    raw_dataset = load_dataset_from_file(train_file)
    print(f"加载了{len(raw_dataset)}个训练样本")
    
    # 标记化数据集
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 设置训练参数，根据模型类型调整
    if model_type == "begin_url":
        # 摘要生成任务
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=os.path.join(model_dir, "logs"),
            fp16=fp16,  # 使用传入的fp16参数
            bf16=bf16,  # 使用传入的bf16参数
            gradient_accumulation_steps=4,
            logging_steps=100,
            save_total_limit=2,
        )
    else:  # end_url
        # 类别预测任务
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate * 1.5,  # 稍微更高的学习率
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=os.path.join(model_dir, "logs"),
            fp16=fp16,  # 使用传入的fp16参数
            bf16=bf16,  # 使用传入的bf16参数
            gradient_accumulation_steps=2,
            logging_steps=100,
            save_total_limit=2,
        )
    
    print(f"训练配置: fp16={fp16}, bf16={bf16}")
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 训练模型
    print("开始训练...")
    trainer.train()
    print("训练完成!")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"模型保存到 {final_model_path}")
    
    return final_model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练begin_url或end_url模型")
    parser.add_argument("--model_type", type=str, choices=["begin_url", "end_url"], required=True, 
                        help="模型类型: begin_url用于摘要生成, end_url用于类别预测")
    parser.add_argument("--train_file", type=str, required=True, 
                        help="训练数据文件路径")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--model_name", type=str, 
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                        help="预训练模型名称或路径")
    parser.add_argument("--fp16", action="store_true", help="使用FP16混合精度训练")
    parser.add_argument("--bf16", action="store_true", help="使用BF16混合精度训练")
    
    args = parser.parse_args()
    
    # 训练模型
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