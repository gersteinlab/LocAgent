import argparse
import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description='Train a language model using Unsloth')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen2.5-Coder-7B-Instruct",
                      help='Name or path of the pretrained model')
    parser.add_argument('--max_seq_length', type=int, default=131072,
                      help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                      help='Use 4-bit quantization')
    
    # Training configuration
    parser.add_argument('--data-path', type=str,
                      help='Path to training data JSONL file')
    parser.add_argument('--output-dir', type=str, default="outputs",
                      help='Path to output directory')
    parser.add_argument('--exp-name', type=str, required=True,
                      help='Name for the output model')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Per device training batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                      help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    # parser.add_argument("--continued_ft", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16,
                      help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=16,
                      help='LoRA alpha parameter')
    parser.add_argument('--warmup_steps', type=int, default=5,
                      help='warmup steps')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    # save args to json
    with open(os.path.join(args.output_dir, args.exp_name, "args.json"), "w") as f:
        json.dump(args.__dict__, f)
    
    # Model initialization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=args.load_in_4bit,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    # Data loading
    with open(args.data_path) as f:
        dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} samples from {args.data_path}")
    dataset = [D["messages"] for D in dataset]
    dataset = Dataset.from_dict({"conversations": dataset})
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Model configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r, # 16
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha, # 16
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=os.path.join(args.output_dir, args.exp_name),
            report_to="wandb",
            run_name=args.exp_name,
            save_strategy="epoch",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Training stats and execution
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    if args.resume_from_checkpoint:
        trainer_stats = trainer.train(resume_from_checkpoint = True)
    else:
        trainer_stats = trainer.train( )

    # Final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save models
    model.save_pretrained(os.path.join(args.output_dir, args.exp_name, "adapter"))
    tokenizer.save_pretrained(
        os.path.join(args.output_dir, args.exp_name, "adapter")
    )
    model.save_pretrained_merged(
        os.path.join(args.output_dir, args.exp_name, f"merged"),
        tokenizer,
        save_method="merged_16bit",
    )

if __name__ == "__main__":
    main()
