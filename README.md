<<<<<<< HEAD
# GRPO_PROJECT_FOR_AUTODL
GRPO-based Mathematical Reasoning Model
=======
# GRPO 数学推理训练手册（AutoDL·离线可复现版）

> - **W&B 离线**（`WANDB_MODE=offline`，不使用 `WANDB_DISABLED`）
>- 版本下限：**PyTorch ≥ 2.6**、`transformers ≥ 4.45`、`accelerate ≥ 0.34`、`trl ≥ 0.17.0`、`datasets ≥ 2.19`、`peft ≥ 0.14`，以及 `safetensors/sentencepiece/einops`
> - **全部缓存与输出走数据盘**（避免系统盘爆满）
> - 模型：`Qwen/Qwen2.5-0.5B-Instruct`；数据集：`openai/gsm8k`
> - 不包含你项目的两个核心文件源码（`reward_funcs.py`、`train_grpo.py`），但**明确它们的放置位置与运行时机/指令**

------

## 0. 目录

- 数据盘（你机器上常见）：`/root/autodl-tmp`
- 工作区（本项目）：`/root/autodl-tmp/projects/grpo-qwen-gsm8k`（简称 **$WORKSPACE**）
- 你已在容器中拥有 `root` 权限，并能访问 GPU（NVIDIA 驱动 OK）。

------

## 1) 一次性环境变量（把缓存全指向数据盘 & W&B 离线）

```bash
# 设定数据盘与缓存
export DATA_DISK="/root/autodl-tmp"
mkdir -p $DATA_DISK/{projects,outputs,.cache/{huggingface/{hub,datasets},pip,wandb}}

# Hugging Face 缓存
export HF_HOME="$DATA_DISK/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$DATA_DISK/.cache/huggingface/datasets"

# pip 缓存
export PIP_CACHE_DIR="$DATA_DISK/.cache/pip"

# W&B 仅离线（不要设置 WANDB_DISABLED）
export WANDB_MODE="offline"
export WANDB_DIR="$DATA_DISK/.cache/wandb"

# （可选）HF 国内镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 项目根目录
export WORKSPACE="$DATA_DISK/projects/grpo-qwen-gsm8k"
mkdir -p "$WORKSPACE" "$DATA_DISK/outputs"
cd "$WORKSPACE"
```

> 建议把以上块追加进 `~/.bashrc`，以后新终端自动生效：
>  `cat >> ~/.bashrc <<'EOF'` …（同上变量块）… `EOF`，再 `source ~/.bashrc`。

------

## 2) 虚拟环境 + 核心依赖安装

### 2.1 用 venv（或你偏好的 conda，二选一）

**venv：**

```bash
python3 -m venv ~/grpo-venv
source ~/grpo-venv/bin/activate
python -m pip install -U pip wheel setuptools
```

（如果你更习惯 **conda**：`conda create -y -n grpo python=3.10 && conda activate grpo`）

### 2.2 安装 PyTorch（针对 **CUDA 12.8 驱动**）

PyTorch 官方 wheel 最高 **cu124**；你的 12.8 驱动可向下兼容 **cu124**：

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch>=2.6.0" torchvision --upgrade
```

**快速自检：**

```bash
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

### 2.3 安装其余依赖

```bash
pip install \
  "transformers>=4.45" \
  "accelerate>=0.34" \
  "trl>=0.17.0" \
  "datasets>=2.19" \
  "peft>=0.14" \
  huggingface_hub safetensors sentencepiece einops \
  "wandb>=0.17" modelscope
```

------

## 3) 准备模型与数据集（缓存都在数据盘）

```bash
# 确认变量生效
source ~/.bashrc
cd "$WORKSPACE"

# 下载模型（保存在本地目录；多次中断可续传）
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="./Qwen2.5-0.5B-Instruct",
    local_dir_use_symlinks=False,
    resume_download=True
)
print("✅ 模型就绪：Qwen2.5-0.5B-Instruct")
PY

# 下载数据集（HF_DATASETS_CACHE 已指向数据盘）
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", cache_dir=None)
print("✅ gsm8k 就绪，train size =", len(ds["train"]))
PY
```

------

## 4) 放置源码文件（不贴代码，只说位置与要点）

> **你需要准备两个文件**，放在 **$WORKSPACE** 根目录：
>
> - `reward_funcs.py`：**奖励函数集合**（正确性、整数、格式、XML 标签完整度等）
>
>   - **兼容新版 TRL**：`completions` 的入参是 `List[str]`（不是 `List[dict]`），奖励函数里直接把每个 `c` 当字符串解析即可。
>   - `<answer>...</answer>` 的抽取要稳健；`gsm8k` 标准答案需提取 `####` 后的数值做比对。
>
> - `train_grpo.py`：**训练入口脚本**
>
>   - **数据预处理**：把 `gsm8k` 的 `question` → `prompt`；并保留 `answer`（取 `####` 后部分）。
>
>   - **强制输出格式**：用 `tokenizer.apply_chat_template` 构造带 **SYSTEM_PROMPT** 的聊天模板，要求模型按
>
>     ```
>     <reasoning>...</reasoning>
>     <answer>\boxed{...}</answer>
>     ```
>
>     输出；这能显著减少“全 0 奖励”的问题。
>
>   - **模型加载**：`dtype=torch.bfloat16`（注意：`torch_dtype` 已弃用，用 `dtype`）。
>
>   - **GRPOConfig 关键点**：
>
>     - 不要使用不支持的参数（如某些版本没有 `vllm_device`，没有 `do_sample` 字段）
>     - 设置 `max_completion_length≈192`（减少被硬截断）
>     - `temperature=0.7, top_p=0.9`（采样出更丰富的答案以便 GRPO）
>     - `report_to="wandb"`（配合环境 `WANDB_MODE=offline`，**不要**设 `WANDB_DISABLED`）
>     - （省空间可选）`save_total_limit=1`、适当调大 `save_steps`
>
>   - **Trainer**：把 `reward_funcs.py` 里各函数装入 `reward_funcs=[...]`
>
>   - 训练结束后保存到 `outputs/Qwen2.5-0.5B-reasoning-GRPO`

**文件所在目录结构建议：**

```
/root/autodl-tmp/projects/grpo-qwen-gsm8k/
│
├── Qwen2.5-0.5B-Instruct/            # 底模（已下载）
├── outputs/                          # 训练输出（自动生成）
│
├── reward_funcs.py                   # 奖励函数（不在此文贴源码）
├── train_grpo.py                     # 训练主脚本（不在此文贴源码）
└── ...
```

------

## 5) 启动训练（第五步）

> **运行时机**：当且仅当你已完成第 1–4 步（环境/依赖/模型/数据 + 两个脚本就位）。

**从头开始训练：**

```bash
cd "$WORKSPACE"
python train_grpo.py
```

**断点重训 & 省空间建议：**

- 若只想保留**一个**最近 checkpoint：在 `GRPOConfig` 里设置 `save_total_limit=1`；
- 断点续训：加 `resume_from_checkpoint=True`（或保持默认行为，很多版本会自动从 `output_dir` 下最新版续起）。

**训练过程观察要点：**

- 一开始就看到 `completions/clipped_ratio < 1.0`、`rewards/* > 0`、`loss`/`grad_norm` 非零 → 正常；
- 如果出现**全为 0**，多半是：
  - 未使用聊天模板/系统提示，导致模型不产 `<answer>`；
  - 奖励函数仍按旧结构读取 `completions`（不是 `List[str]`）；
  - `max_completion_length` 太小/采样设置不当。

------

## 6) 数据盘空间管理（爆盘处理与预防）

**清理旧 checkpoint，仅保留最新：**

```bash
cd $WORKSPACE/outputs/Qwen2.5-0.5B-reasoning-GRPO
ls -dt checkpoint-* | tail -n +2 | xargs -r rm -rf
```

**清理 HF 数据集缓存（可再下）：**

```bash
rm -rf /root/autodl-tmp/.cache/huggingface/datasets/*
```

**清理 pip/wandb 缓存：**

```bash
rm -rf /root/autodl-tmp/.cache/pip/*
rm -rf /root/autodl-tmp/.cache/wandb/*   # 若你已打包/同步日志
```

**查看盘占用：**

```bash
df -h
du -h --max-depth=1 /root/autodl-tmp | sort -h
```

> **注意**：盘满保存中断会导致 checkpoint **损坏**（`safetensors ... incomplete metadata`）。遇到时直接换用上一个完整的 checkpoint。

------

## 7) 推理测试（不贴源码，只给用法要点）

- **加载 checkpoint 模型**推理时，**tokenizer 请用底模目录**（checkpoint 里通常不含 tokenizer 文件）：
  - `AutoTokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct")`
  - `AutoModelForCausalLM.from_pretrained("outputs/.../checkpoint-XXXX", dtype=torch.bfloat16, device_map="auto")`
- **构造 prompt**时与训练保持一致（同样的 `SYSTEM_PROMPT` + `apply_chat_template`），再 `generate(max_new_tokens≈192)`。
- 如某个 checkpoint 无法加载（损坏），请换用上一个。

------

## 8) W&B 离线日志查看/上传

### 8.1 在容器内本地查看（如果网络限制无法上云）

```bash
# 最新离线 run 目录
ls -dt /root/autodl-tmp/.cache/wandb/wandb/offline-run-* | head -n1

# 在该目录里开本地可视化（容器内）
wandb local
```

### 8.2 离线日志打包带回本地再上传（推荐）

**在容器内打包（排除 symlink 的 logs 目录，避免解压报错）：**

```bash
cd /root/autodl-tmp/.cache/wandb/wandb
LATEST=$(ls -dt offline-run-* | head -n1)
tar --exclude='offline-run-*/logs' -czf latest_wandb_offline_run.tar.gz "$LATEST"
```

把 `latest_wandb_offline_run.tar.gz` 下载到你的本地电脑。

**在本地新建干净 venv 专用同步：**

```bat
# Windows PowerShell / CMD
python -m venv E:\wandb_env
E:\wandb_env\Scripts\activate
pip install "wandb" "pydantic<2"
wandb login   # 输入你的 API Key

# 解压并同步
tar -xzf latest_wandb_offline_run.tar.gz
wandb sync offline-run-XXXXXXXX_XXXXXX-XXXXXXXX
```

> 本地如果解压遇到 symlink 报错，可用：
>  `tar --no-same-owner --ignore-failed-read -xzf latest_wandb_offline_run.tar.gz`

------

## 9) 常见问题速查（都是这次踩过的点）

- **`KeyError: 'prompt'`**：`GRPOTrainer` 需要字段名 `prompt`；记得把 `question → prompt` 并保留 `answer`。
- **奖励全 0 / grad_norm=0**：未使用聊天模板+系统提示，或奖励函数按旧结构读取 `completions`。
- **`string indices must be integers`**：新版 TRL 的 `completions` 是 `List[str]`。
- **`... unexpected keyword argument 'vllm_device' / 'do_sample'`**：你的 TRL 版本不支持这些参数——删掉即可。
- **`WANDB_DISABLED` 冲突**：只用 `WANDB_MODE=offline`，不要设置 `WANDB_DISABLED`。
- **`safetensors ... incomplete metadata`**：checkpoint 半途写入（磁盘满），换上一个完整的 checkpoint。
- **`transformers` 缓存变量弃用**：看到 `TRANSFORMERS_CACHE` 的 deprecation 警告即可，无伤；我们已用 `HF_HOME`。
- **CUDA 12.8 驱动**：安装 PyTorch 的 **cu124** 轮子，能正常工作。
- **找不到 tokenizer**：加载 checkpoint 推理时，tokenizer 请仍指向底模目录。
- **路径没加引号**：诸如 `model_dir = "outputs/.../checkpoint-1700"` 必须是**字符串**。

------

## 11) 最后：两个脚本的**运行时机与指令**

- **`reward_funcs.py`**：只需放置在 **$WORKSPACE** 根目录，**不需要单独运行**；被训练脚本 `import` 使用。

- **`train_grpo.py`**：当 1–4 步完成后，执行 **第五步**启动训练：

  ```bash
  cd /root/autodl-tmp/projects/grpo-qwen-gsm8k
  python train_grpo.py
  ```

  - **重新训练（从头）**：先删除旧输出目录

    ```bash
    rm -rf /root/autodl-tmp/projects/grpo-qwen-gsm8k/outputs/Qwen2.5-0.5B-reasoning-GRPO
    python train_grpo.py
    ```

  - **断点续训**：保留 `outputs/`，在 `GRPOConfig` 中（或默认行为）自动从最新 checkpoint 续起。
>>>>>>> e1d5851 (grpo project first commit)
