import torch
import torch.nn as nn
from transformers import AutoModel

# 1. 声明高维解耦头
class DecoupledRewardHead(nn.Module):
    def __init__(self, hidden_dim=2048, up_dim=16384):
        super().__init__()
        self.w_up = nn.Linear(hidden_dim, up_dim, bias=False)
        self.act = nn.GELU()
        self.w_score = nn.Linear(up_dim, 1, bias=False)      
        # 初始化
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_score.weight, std=0.02)

    def forward(self, x):
        return self.w_score(self.w_up(x))

# 2. 强制加载你本地刚下载好的物理仓库
model_path = "./internlm2_reward"
print("[开始] 正在从本地拉取 InternLM2 原生奖励模型...")
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

# 3. 精准切除原厂的 v_head，换上我们设计的 Decoupled 架构
hidden_size = model.config.hidden_size # 1.8B 模型自动获取为 2048
print(f"[识别] 原厂模型隐层维度 hidden_size = {hidden_size}")

# 偷梁换柱
model.v_head = DecoupledRewardHead(hidden_dim=hidden_size, up_dim=16384).to(model.device).half()
print("[注入] 成功！原厂单层线性 v_head 已被 $W_{up}$ 高维解耦空间替代。")

# 4. 执行 Freeze 策略：锁死全身体外组织，只给新脑建立梯度传导
for param in model.model.parameters(): # 冻结底座的 Transformer 24层
    param.requires_grad = False

for param in model.v_head.parameters(): # 仅放开你新写的高维头
    param.requires_grad = True

# 5. 检查可训练参数验证算力开销
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[验证] 当前可训练参数量: {trainable_params / 1e6:.2f} M")
print("结构改造彻底就绪。接下来我们将模拟一轮偏好对齐训练，验证计算图是否闭环，奖励分数是否合理提升。")