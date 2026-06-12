import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. 注入环境
model_path = "./internlm2_reward"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, dtype=torch.float16, trust_remote_code=True)

# 2. 依然插入W_up 架构
class DecoupledRewardHead(nn.Module):
    def __init__(self, hidden_dim=2048, up_dim=16384):
        super().__init__()
        self.w_up = nn.Linear(hidden_dim, up_dim, bias=False)
        self.act = nn.GELU()
        self.w_score = nn.Linear(up_dim, 1, bias=False)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_score.weight, std=0.02)

    def forward(self, x):
        return self.w_score(self.act(self.w_up(x)))

model.v_head = DecoupledRewardHead(hidden_dim=model.config.hidden_size, up_dim=16384)

for param in model.model.parameters(): param.requires_grad = False
for param in model.v_head.parameters(): param.requires_grad = True

# 将模型推向 Mac Studio 的图形核心（Metal 架构，mps 或者是 cpu 运行 float32）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
if device.type == "cpu":
    model = model.float() # CPU 上跑 float32 更稳定

# 3. 构造一个经典的金融/逻辑对齐的“模拟假数据对”来测试计算图是否闭环
prompt = "公司的流动比率很高，说明了什么？"
chosen_ans = "说明短期偿债能力强，但资产利用效率可能不高。"
rejected_ans = "说明公司快要破产了，需要赶紧抛售股票。"
chosen_text = f"<s><|User|>\n{prompt}<|Bot|>\n{chosen_ans}</s>"
rejected_text = f"<s><|User|>\n{prompt}<|Bot|>\n{rejected_ans}</s>"

# 4. 实例化优化器
optimizer = torch.optim.AdamW(model.v_head.parameters(), lr=1e-4)

print("\n[开始] 模拟一轮偏好对齐训练...")

# 5. 前向传播提取特征
chosen_inputs = tokenizer(chosen_text, return_tensors="pt").to(device)
rejected_inputs = tokenizer(rejected_text, return_tensors="pt").to(device)

# 得到原厂底座提取出来的隐特征 [1, seq_len, 2048]
chosen_outputs = model.model(**chosen_inputs)
rejected_outputs = model.model(**rejected_inputs)

# 提取最后一个 Token（即代表整句奖励池的特殊标记）的隐状态
# 形状从 [1, seq_len, 2048] 精准切片为 [1, 2048]
chosen_last_feature = chosen_outputs.last_hidden_state[:, -1, :]
rejected_last_feature = rejected_outputs.last_hidden_state[:, -1, :]

# 将切片后的干净特征送入你的 $W_{up}$ 高维解耦空间
chosen_score = model.v_head(chosen_last_feature)   # 此时形状输出为标准的 [1, 1]
rejected_score = model.v_head(rejected_last_feature) # 此时形状输出为标准的 [1, 1]

print(f"-> 训练前原始打分 - Chosen(好): {chosen_score.item():.4f} | Rejected(坏): {rejected_score.item():.4f}")

# 6. 计算标准 Pairwise Ranking Loss
loss = -torch.log(torch.sigmoid(chosen_score - rejected_score)).mean()
print(f"-> 当前对齐 Loss: {loss.item():.4f}")

# 7. 反向传播与梯度更新
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("[成功] 梯度反向传播完成！33M 参数已被修正。")

# 再次验证得分变化
with torch.no_grad():
    new_chosen_feature = model.model(**chosen_inputs).last_hidden_state[:, -1, :]
    new_rejected_feature = model.model(**rejected_inputs).last_hidden_state[:, -1, :]
    
    new_chosen = model.v_head(new_chosen_feature)
    new_rejected = model.v_head(new_rejected_feature)
    
    print(f"-> 训练后更新打分 - Chosen(好): {new_chosen.item():.4f} | Rejected(坏): {new_rejected.item():.4f}")