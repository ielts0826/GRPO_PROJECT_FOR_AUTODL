import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# 导入奖励函数
from reward_funcs import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)

# ============ 系统提示模板 ============
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
\\boxed{...}
</answer>
End your reply right after </answer> with nothing else."""

# ============ 模型与数据 ============
model_name = "./Qwen2.5-0.5B-Instruct"  # 本地模型目录
raw_dataset = load_dataset("openai/gsm8k", "main")["train"]

# 数据预处理：构造 prompt + 提取标准答案
def preprocess(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    gold = example["answer"].split("####")[-1].strip()
    return {"prompt": prompt_text, "answer": gold}

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,   # torch_dtype 已弃用，换成 dtype
    device_map="auto"
)

# ============ GRPO 训练参数 ============
training_args = GRPOConfig(
    output_dir="outputs/Qwen2.5-0.5B-reasoning-GRPO",  # 输出路径
    run_name="Qwen2.5-0.5B-GRPO-gsm8k",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,  # 使用 bfloat16 混合精度
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=8,       # 每个问题生成多个答案
    max_prompt_length=256,
    max_completion_length=192,   # 略小于 200，减少硬截断
    temperature=0.7,
    top_p=0.9,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    report_to="wandb"        # 设为 offline，不会联网
)

# ============ Trainer ============
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
)

# ============ 启动训练 ============
trainer.train()
trainer.save_model("outputs/Qwen2.5-0.5B-reasoning-GRPO")
