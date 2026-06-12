import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. 注入环境
model_path = "./internlm2_reward"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, dtype=torch.float16, trust_remote_code=True)

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

# 2. 插入纯线性 W_up 架构
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

for param in model.model.parameters(): param.requires_grad = False
for param in model.v_head.parameters(): param.requires_grad = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
if device.type == "cpu":
    model = model.float() 

# 3. 模拟假数据对
prompt = "公司的流动比率很高，说明了什么？"
chosen_ans = "说明短期偿债能力强，但资产利用效率可能不高。"
rejected_ans = "说明公司快要破产了，需要赶紧抛售股票。"
chosen_text = f"<s><|User|>\n{prompt}<|Bot|>\n{chosen_ans}</s>"
rejected_text = f"<s><|User|>\n{prompt}<|Bot|>\n{rejected_ans}</s>"

optimizer = torch.optim.AdamW(model.v_head.parameters(), lr=1e-4)
print("\n[开始] 模拟一轮纯线性高维偏好对齐训练...")

chosen_inputs = tokenizer(chosen_text, return_tensors="pt").to(device)
rejected_inputs = tokenizer(rejected_text, return_tensors="pt").to(device)

chosen_outputs = model.model(**chosen_inputs)
rejected_outputs = model.model(**rejected_inputs)

chosen_last_feature = chosen_outputs.last_hidden_state[:, -1, :]
rejected_last_feature = rejected_outputs.last_hidden_state[:, -1, :]

chosen_score = model.v_head(chosen_last_feature)   
rejected_score = model.v_head(rejected_last_feature) 

print(f"-> 训练前原始打分 - Chosen(好): {chosen_score.item():.4f} | Rejected(坏): {rejected_score.item():.4f}")

loss = -torch.log(torch.sigmoid(chosen_score - rejected_score) + 1e-6).mean()
print(f"-> 当前对齐 Loss: {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("[成功] 33M 参数纯线性映射已被修正。")
