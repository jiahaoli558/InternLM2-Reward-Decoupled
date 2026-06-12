import os
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# 1. 基础环境配置
model_path = "./internlm2_reward"
save_path = "./decoupled_head.pt"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[配置] 当前计算设备: {device}")

# 2. 初始化分词器与底座
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_dtype = torch.float32 if device.type == "cpu" else torch.float16
model = AutoModel.from_pretrained(model_path, torch_dtype=model_dtype, trust_remote_code=True)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        denom = torch.rsqrt(variance + self.eps)
        return (x_f32 * denom).to(input_dtype) * self.weight
    
# 3. 注入纯线性 $W_{up}$ 高维解耦架构
class DecoupledRewardHead(nn.Module):
    def __init__(self, hidden_dim=2048, up_dim=16384):
        super().__init__()
        self.up_dim = up_dim
        self.rms1 = RMSNorm(hidden_dim, eps=1e-6) 
        self.w_up = nn.Linear(hidden_dim, up_dim, bias=False)
        self.rms2 = RMSNorm(up_dim, eps=1e-6) 
        self.w_score = nn.Linear(up_dim, 1, bias=False)
        
        nn.init.orthogonal_(self.w_up.weight, gain=0.01)
        nn.init.normal_(self.w_score.weight, std=0.00001)

    def forward(self, x):
        x = self.rms1(x)
        x = self.w_up(x)
        x = self.rms2(x)
        score = self.w_score(x) / (self.up_dim ** 0.5)
        return torch.clamp(score, min=-10.0, max=10.0)

model.v_head = DecoupledRewardHead(hidden_dim=model.config.hidden_size, up_dim=16384)
if device.type != "cpu":
    model.v_head = model.v_head.half()

model = model.to(device)

# 4. 固化底座
for param in model.model.parameters(): 
    param.requires_grad = False
for param in model.v_head.parameters(): 
    param.requires_grad = True

# 5. 加载偏好数据集
print("\n[数据] 正在拉取真实偏好数据集 ...")
raw_train = load_dataset("Baidicoot/anthropic-hh-rlhf", split="train").shuffle(seed=42).select(range(1000))
raw_test = load_dataset("Baidicoot/anthropic-hh-rlhf", split="test").shuffle(seed=42).select(range(200))

class HFPreferenceDataset(Dataset):
    def __init__(self, dataset):
        self.pairs = [{"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"]} for item in dataset]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

train_loader = DataLoader(HFPreferenceDataset(raw_train), batch_size=4, shuffle=True)
test_pool = HFPreferenceDataset(raw_test)

def extract_last_valid_feature(last_hidden_state, attention_mask):
    last_valid_indices = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.size(0)
    return last_hidden_state[torch.arange(batch_size), last_valid_indices, :]

optimizer = torch.optim.AdamW(model.v_head.parameters(), lr=1e-4, weight_decay=0.1, eps=1e-6)

print("\n=== [阶段一] 开始 1000 条数据流的高维探针微调 ===")
model.train()

for epoch in range(1):
    for batch_idx, batch in enumerate(train_loader):
        chosen_texts = [f"<s><|User|>\n{p}<|Bot|>\n{c}</s>" for p, c in zip(batch["prompt"], batch["chosen"])]
        rejected_texts = [f"<s><|User|>\n{p}<|Bot|>\n{r}</s>" for p, r in zip(batch["prompt"], batch["rejected"])]
        
        c_in = tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        r_in = tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        c_outputs = model.model(**c_in)
        r_outputs = model.model(**r_in)
        
        c_feat = extract_last_valid_feature(c_outputs.last_hidden_state, c_in["attention_mask"]).contiguous()
        r_feat = extract_last_valid_feature(r_outputs.last_hidden_state, r_in["attention_mask"]).contiguous()
        
        c_score = model.v_head(c_feat)
        r_score = model.v_head(r_feat)
        
        # 标准的 Pairwise Ranking 损失
        loss = -torch.log(torch.sigmoid(c_score - r_score) + 1e-6).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.v_head.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx:03d}/{len(train_loader)} | Loss: {loss.item():.4f} | 平均分差 (Margin): {(c_score - r_score).mean().item():.2f}")

print("\n=== [阶段二] 固化并保存全新高维 Head 参数 ===")
torch.save(model.v_head.state_dict(), save_path)

print("\n=== [阶段三] 重新载入保存权重，进行随机未知样本的泛化测试 ===")
model.v_head.load_state_dict(torch.load(save_path))
model.eval()

random_samples = random.sample(list(test_pool), 3)

for idx, sample in enumerate(random_samples):
    print(f"\n----------------- 随机泛化测试样本 {idx+1} -----------------")
    c_text = f"<s><|User|>\n{sample['prompt']}<|Bot|>\n{sample['chosen']}</s>"
    r_text = f"<s><|User|>\n{sample['prompt']}<|Bot|>\n{sample['rejected']}</s>"
    
    c_in = tokenizer(c_text, return_tensors="pt").to(device)
    r_in = tokenizer(r_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        c_outputs = model.model(**c_in)
        r_outputs = model.model(**r_in)
        
        # 测试推理也全面采用动态 Token 提取机制，完全消除 PAD 噪声
        c_feat = extract_last_valid_feature(c_outputs.last_hidden_state, c_in["attention_mask"])
        r_feat = extract_last_valid_feature(r_outputs.last_hidden_state, r_in["attention_mask"])
        
        final_chosen_score = model.v_head(c_feat).item()
        final_rejected_score = model.v_head(r_feat).item()
        
    margin = final_chosen_score - final_rejected_score
    print(f"-> 固化权重打分 - Chosen: {final_chosen_score:.4f} | Rejected: {final_rejected_score:.4f}")
    print(f"-> 泛化排序代差 (Margin): {margin:.4f} | 是否符合人类偏好: {margin > 0}")
