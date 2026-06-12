import torch
import torch.nn as nn
from transformers import AutoModel

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

# 1. 声明纯线性高维解耦头
class DecoupledRewardHead(nn.Module):
    def __init__(self, hidden_dim=2048, up_dim=16384):
        super().__init__()
        self.up_dim = up_dim
        self.rms1 = RMSNorm(hidden_dim, eps=1e-6) 
        self.w_up = nn.Linear(hidden_dim, up_dim, bias=False)
        self.rms2 = RMSNorm(up_dim, eps=1e-6) 
        self.w_score = nn.Linear(up_dim, 1, bias=False)
        
        # 严格初始化控制
        nn.init.orthogonal_(self.w_up.weight, gain=0.01)
        nn.init.normal_(self.w_score.weight, std=0.00001)

    def forward(self, x):
        x = self.rms1(x)
        x = self.w_up(x)   # 纯线性暴力投影解耦
        x = self.rms2(x)
        score = self.w_score(x) / (self.up_dim ** 0.5)
        return torch.clamp(score, min=-10.0, max=10.0)

# 2. 强制加载你本地刚下载好的物理仓库
model_path = "./internlm2_reward"
print("[开始] 正在从本地拉取 InternLM2 原生奖励模型...")
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

# 3. 精准切除原厂的 v_head，换上我们设计的 Decoupled 架构
hidden_size = model.config.hidden_size 
print(f"[识别] 原厂模型隐层维度 hidden_size = {hidden_size}")

# 偷梁换柱
model.v_head = DecoupledRewardHead(hidden_dim=hidden_size, up_dim=16384).to(model.device).half()
print("[注入] 成功！原厂单层线性 v_head 已被 $W_{up}$ 纯线性高维解耦空间替代。")

# 4. 执行 Freeze 策略
for param in model.model.parameters(): 
    param.requires_grad = False
for param in model.v_head.parameters(): 
    param.requires_grad = True

# 5. 检查可训练参数验证算力开销
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[验证] 当前可训练参数量: {trainable_params / 1e6:.2f} M")
print("结构改造彻底就绪。")
