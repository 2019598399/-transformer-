import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from model import *
import matplotlib.pyplot as plt
import math
import json
from transformers import AutoConfig
from device import *


#路径，修改这里的路径可以使用自己的模型或数据集
dataset_path = "./train.jsonl"
model_path = "./Qwen2.5-3B"


class Mydata(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.a_datas = []
        with open(self.root_dir, 'r', encoding="utf-8") as f:
            for line in f:
                a_data = json.loads(line)
                self.a_datas.append(a_data)

    def __getitem__(self, idx):
        return self.a_datas[idx]

    def __len__(self):
        return len(self.a_datas)

train_data = Mydata(dataset_path)
train_data_size = len(train_data)
print(f"训练集的长度为：{train_data_size}")

batch_size = 1
num_train_epochs = 7
gradient_accumulation_steps = 2
learning_rate = 1e-6
bs = batch_size  # 4
accum = gradient_accumulation_steps  # 8
num_update_steps_per_epoch = math.ceil(train_data_size / (bs * accum))
total_steps = num_update_steps_per_epoch * num_train_epochs
num_warmup_steps = int(0.1 * total_steps)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda batch: batch)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：语言模型
    inference_mode=False,          # False 表示训练时插入 LoRA 层
    r=16,                          # LoRA 的秩 (rank)
    lora_alpha=32,                 # LoRA 的缩放系数
    lora_dropout=0.05,             # dropout 概率
    target_modules=["q_proj", "v_proj"]
    # 指定哪些模块插入低秩分解（通常是 q、k、v 投影层）
)

model = MultiTaskModel("./Qwen2.5-3B").to(device)
model.base_model = get_peft_model(model.base_model, lora_config)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

train_loss_x = []
train_loss_y = []

for epoch in range(num_train_epochs):
    total_loss = 0
    model.train()
    step = 0
    for a_data in train_dataloader:
        a_data = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
            for sample in a_data
        ]
        with torch.autograd.set_detect_anomaly(True):
            l = model(a_data)
            total_loss = total_loss + l.item()
            l = l / gradient_accumulation_steps
            l.backward()
        step = step + 1
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.npu.empty_cache()
            print(f"Epoch {epoch + 1} step {step}: loss={l.item() * gradient_accumulation_steps:.4f}")
            if torch.isnan(l):
                print("Found NaN loss at step", step)
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Grad NaN in {name}")
                # 如果发现 NaN，可以选择抛出异常或 break
                raise ValueError("NaN encountered, stop training")
    train_loss_x.append(epoch)
    train_loss_y.append(total_loss)

# 画出训练损失曲线
def plot():
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_x, train_loss_y, label="Lora-finetune", color="blue", linestyle='-', marker='o')
    plt.title("Train Loss Curve", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.savefig("train_loss_curve.png")
    plt.close()

plot()


# 2. 保存合并后的模型
model.base_model = model.base_model.merge_and_unload()
output_dir = "./qwen2.5-3b-merged"
model.base_model.save_pretrained(output_dir)

# 3. 保存 tokenizer 和 config（确保加载时与原始模型一致）
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.save_pretrained(output_dir)
config = AutoConfig.from_pretrained(model_path)
config.save_pretrained(output_dir)

# 检查合并后的模型和配置
print(f"Model saved to: {output_dir}")
