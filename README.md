# InternLM2-Reward-Decoupled

高维解耦轻量化奖励模型微调方案。本仓库分享一个针对 **InternLM2-Reward (1.8B/7B)** 的原生奖励头（`v_head`）进行重构的实验方案。

传统奖励模型（RM）通常使用单层线性变换将基座模型的最后隐层状态（如 2048 维）直接压缩为 1 维标量分数。这种极端的线性压缩可能导致奖励流形在低维空间发生严重塌陷与语义退化。

为了打破这种传统的机械变换，本方案构想并实现了一个**高维解耦判官头（Decoupled Reward Head）**，将隐状态拉升至高维超空间进行流形解耦与自旋过滤，并在仅放开 33M 参数的极小开销下，在 Mac Studio (Apple Silicon) 及主流 GPU 设备上完成了完整的偏好对齐微调闭环。

---

## 核心设计思想

### 1. 高维流形解耦 ($W_{up}$)
我们设计了 `DecoupledRewardHead` 替换原厂单层线性 `v_head`。首先通过一个极大的升维矩阵 $W_{up}$，将 2048 维的隐状态暴力拉升到 **16384 维**的高维流形空间。在高维超空间中，不同维度的正交性大幅增强，各种复杂的语义偏好特征得以在更加开阔的基底上实现真正的空间解耦。

### 2. 双重 RMSNorm 锁死方差与稳定性保险
在高维空间中，特征矩阵相乘极易引发数值爆炸（Exploding）或下溢，导致半精度（FP16/BF16）训练发生严重的梯度断流。为此我们引入了多重物理保险：
* **进入超空间前**：经过第一层 `RMSNorm` 规范化基座输出。
* **流形扭曲后**：在 16384 维超空间中应用第二层 `RMSNorm` 锁死方差。
* **缩放点积机制**：在输出最终标量时，效仿 Attention 思想，将分数除以 $\sqrt{d_{up}}$（即 $\sqrt{16384}$），彻底断绝规模爆炸，并将最终得分合理限幅（Clamp）在 $[-10.0, 10.0]$ 内。

### 3. 动态索引拒绝 PAD 噪声污染
原厂在处理 Batch 文本时常受到 Padding Token 的干扰。本实现摒弃了粗暴的 `[:, -1, :]` 切片，编写了 `extract_last_valid_feature` 函数，根据 `attention_mask` 动态提取每个样本真正合法的最后一个有效 Token 特征，确保注入解耦空间的信号绝对纯净。

---

## 架构核心代码

```python
import torch
import torch.nn as nn

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

class DecoupledRewardHead(nn.Module):
    def __init__(self, hidden_dim=2048, up_dim=16384):
        super().__init__()
        self.up_dim = up_dim
        
        self.rms1 = RMSNorm(hidden_dim, eps=1e-6) 
        self.w_up = nn.Linear(hidden_dim, up_dim, bias=False)
        self.act = nn.Tanh()
        self.rms2 = RMSNorm(up_dim, eps=1e-6) 
        self.w_score = nn.Linear(up_dim, 1, bias=False)
        
        # 谨慎初始化：正交初始化控益，标量输出压制标准差
        nn.init.orthogonal_(self.w_up.weight, gain=0.01)
        nn.init.normal_(self.w_score.weight, std=0.00001)

    def forward(self, x):
        x = self.rms1(x)
        x = self.w_up(x)
        # x = self.act(x) # 可选：引入非线性激活空间流形扭曲
        x = self.rms2(x)
        
        # 【核心修正】：除以 sqrt(up_dim) 抵消高维空间相乘的规模爆炸
        score = self.w_score(x) / (self.up_dim ** 0.5)
        return torch.clamp(score, min=-10.0, max=10.0)

```

---

## 微调与轻量化策略

我们执行了严格的 **Freeze 策略**：锁死全身体外组织（冻结底座 Transformer 的所有 24 层），只给新设计的 `DecoupledRewardHead` 建立梯度传导。

* **底座参数**：InternLM2-1.8B (全部冻结)
* **可训练参数量**：**32.77 M** (仅占全量参数的极小部分)
* **算力开销**：由于固定了底座，微调时无需计算庞大的 Transformer 全量梯度。在配备 36GB 统一内存的 Mac Studio (MPS 架构) 或消费级单卡 GPU 上，即可全速处理工业级偏好数据流。

### 动态微调计算图

```text
[输入序列 Chosen/Rejected] 
       │
[InternLM2 Base (Frozen)] 
       │ ──> 经 attention_mask 动态提取有效尾端特征
[Last Valid Hidden State (2048维)] 
       │
[RMSNorm 1]
       │
[W_up 暴力投影 (16384维超空间)] ──> 语义特征高维解耦
       │
[RMSNorm 2]
       │
[W_score 映射 & 根号缩放]
       │
[Scalar Score (标量分数)] ──> Pairwise Ranking Loss ──> 仅反向传播修正 33M 参数

```

---

## 运行实验与泛化验证

本方案在 `Baidicoot/anthropic-hh-rlhf` 中文偏好数据集上进行了高强度微调测试。

* **零钱模拟对齐**：使用逻辑/金融对齐数据（如流动比率分析问答）进行闭环测试，计算图完美流转，训练后 Chosen 项与 Rejected 项的分差（Margin）明显拉开。
* **真实流微调**：在 1000 条真实偏好流上迭代，配合 AdamW 优化器与梯度裁剪（max_norm=1.0），Loss 稳步收敛，平均分差（Margin）保持健康拉正。
* **纯净泛化测试**：在从未参与过训练的测试池中随机抽取样本，固化权重后的探针网络依然能精准对未知样本输出符合人类偏好（Margin > 0）的正确排序。

---

## 如何运行

1. 下载 InternLM2 奖励模型至本地 `./internlm2_reward`。
2. 运行 `inject_and_train.py` 进行架构重构与验证。
3. 运行 `train_step.py` 进行单步模拟与计算图闭环测试。
4. 运行 `run_experiment.py` 在工业级偏好数据集上执行高强度微调与泛化测试。

```

```
