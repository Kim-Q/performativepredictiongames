# AGDLeader 无参数模型文档

## 概述

`AGDLeader` 是一个**完全基于观测**的领导者控制器，不需要知道 follower 的任何内部参数（d, q, r）。它通过观测 `(state, follower_actions)` 来学习 follower 的响应模型，并据此优化自己的控制策略。

---

## 核心设计思想

### 问题设定

```
┌─────────────────────────────────────────────────────────┐
│                    Env2 环境                              │
│  ẋ = f(x) + g0⊙u0 + g1⊙u1 + g2⊙u2 + g3⊙u3               │
│                                                         │
│  Leader (u0):  ??? 不知道 follower 的任何参数             │
│  Follower i (ui): 有内部参数 (di, qi, ri)                │
│                                                         │
│  Leader 只能观测：state 和 follower_actions              │
└─────────────────────────────────────────────────────────┘
```

### 解决思路

**黑盒建模**：将每个 follower 视为一个未知函数：
```
ui = Fi(state, u0)  ← 用参数化模型近似
```

**线性近似**：使用基函数展开：
```
ui ≈ Φ(state, u0)^T @ Wi
```

其中：
- `Φ(state, u0)` 是已知的基函数向量（12 维）
- `Wi` 是需要学习的权重向量（12 维）

---

## AGDLeader 类结构

```python
class AGDLeader:
    """
    基于观测的领导者 - 完全不知道 follower 的内部参数。
    通过观测 (state, follower_actions) 学习 follower 的响应模型。
    """
```

### 属性说明

| 属性 | 类型 | 说明 |
|------|------|------|
| `env` | Env2 | 环境对象 |
| `num_followers` | int | follower 数量（默认 3） |
| `phi_dim` | int | 基函数维度（12） |
| `W` | List[np.ndarray] | 每个 follower 的响应模型权重 (3×12) |
| `K` | np.ndarray | 控制增益矩阵（暂未使用） |
| `eta_W` | float | 模型学习率（0.01） |
| `eta_K` | float | 控制优化学习率（0.001） |
| `Q0, R0` | np.ndarray | Leader 的代价函数参数 |
| `C` | List[float] | 对 follower 动作的耦合权重 |

### 核心方法

#### 1. 响应模型学习

```python
def update_model(self, state, leader_action, actual_follower_actions, eta=None):
    """
    使用 SGD 更新 follower 响应模型。
    
    对每个 follower i:
    1. 计算预测：u_pred = Φ(state, u0)^T @ W[i]
    2. 计算误差：error = u_actual - u_pred
    3. 梯度下降：W[i] += eta * error * Φ(state, u0)
    """
```

**数学原理**：
```
损失函数：L(W) = ½ ||u_actual - Φ^T @ W||²

梯度：∇L = -Φ @ (u_actual - Φ^T @ W) = -Φ @ error

更新：W_new = W - eta * (-Φ @ error) = W + eta * error * Φ
```

#### 2. 动作预测

```python
def predict_follower_action(self, state, leader_action, i):
    """
    预测 follower i 的动作。
    
    u_i_pred = Φ(state, u0)^T @ W[i]
    """
```

#### 3. 最优控制计算

```python
def compute_optimal_action(self, state):
    """
    基于估计的 follower 模型计算最优控制。
    
    Leader 的代价函数:
    J0 = x^T Q0 x + (u0 + Σ ci * ui)^T R0 (u0 + Σ ci * ui)
    
    使用估计的模型 ui ≈ Φ(state, u0)^T @ W[i]
    
    通过不动点迭代求解最优 u0。
    """
```

#### 4. 单步执行

```python
def step(self, state):
    """观测 → 预测 → 优化 → 行动"""

def learn(self, state, leader_action, follower_actions):
    """学习 follower 响应模型"""
```

---

## 基函数设计

### 当前实现：`phi_cross`

```python
def phi_cross(state, action):
    """
    交叉基函数（12 维）:
    Φ = [x1, x2, u[0], u[1],              # 线性项
         x1*u[0], x1*u[1], x2*u[0], x2*u[1],  # 交叉项
         x1², x2², u[0]², u[1]²]           # 二次项
    """
```

### 基函数选择的影响

| 基函数类型 | 维度 | 优点 | 缺点 |
|-----------|------|------|------|
| 线性 `[x1, x2, u0, u1]` | 4 | 计算快，不易过拟合 | 无法捕捉非线性 |
| 交叉（当前） | 12 | 平衡表达能力和复杂度 | 需要更多数据 |
| 多项式（到 3 阶） | 20+ | 表达能力强 | 易过拟合，计算慢 |
| RBF 核 | 可调 | 万能近似 | 需要选择中心点 |

---

## 训练流程

```python
# 1. 创建环境和 follower
env = Env2()
followers = [FollowerAgent(...), ...]  # 内部参数对 Leader 隐藏

# 2. 创建 Leader（不知道 follower 的任何参数）
leader = AGDLeader(env, num_followers=3)

# 3. 训练循环
for t in range(max_iters):
    state = env.state.copy()
    
    # Leader 行动（基于观测的预测）
    leader_action = leader.step(state)
    
    # Follower 响应（Leader 只能观测结果）
    follower_actions = [f.compute_action(state, leader_action) for f in followers]
    
    # Leader 学习（更新响应模型 W）
    leader.learn(state, leader_action, follower_actions)
    
    # 环境步进
    env.step(leader_action, *follower_actions)
```

---

## 改进方向

### 方向 1：更好的基函数

**当前问题**：`phi_cross` 可能无法充分捕捉 follower 的非线性响应。

**改进方案**：
```python
# 方案 A: 增加高阶项
def phi_poly3(state, action):
    """三阶多项式基函数（20 维）"""
    # ... x1³, x2³, x1²*u0, ...

# 方案 B: RBF 核基函数
def phi_rbf(state, action, centers=None):
    """RBF 基函数"""
    if centers is None:
        centers = np.random.randn(20, 4)  # 20 个中心点
    x = np.concatenate([state, action])
    return np.exp(-np.sum((x - centers)**2, axis=1))
```

### 方向 2：正则化防止过拟合

**当前问题**：SGD 更新可能导致 `W` 过大，过拟合噪声。

**改进方案**：
```python
def update_model(self, state, leader_action, actual_follower_actions, eta=None):
    # ... 原有代码 ...
    
    # 添加 L2 正则化
    for i in range(self.num_followers):
        self.W[i] -= eta * 0.01 * self.W[i]  # 权重衰减
        self.W[i] = np.clip(self.W[i], -10, 10)
```

### 方向 3：自适应学习率

**当前问题**：固定学习率 `eta_W=0.01` 可能不适合所有阶段。

**改进方案**：
```python
# 方案 A: 学习率衰减
self.eta_W = 0.01 / np.log(t + 2)

# 方案 B: Adam 优化器
class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.m = 0  # 一阶矩
        self.v = 0  # 二阶矩
        self.t = 0
    
    def update(self, grad, W):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)
        return W - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
```

### 方向 4：批量更新

**当前问题**：单步 SGD 更新方差大，不稳定。

**改进方案**：
```python
def update_model_batch(self, batch_size=32):
    """使用历史数据批量更新"""
    if len(self.history_states) < batch_size:
        return
    
    # 随机采样 batch
    indices = np.random.choice(len(self.history_states), batch_size, replace=False)
    
    total_grad = [np.zeros(self.phi_dim) for _ in range(self.num_followers)]
    
    for idx in indices:
        state = self.history_states[idx]
        leader_action = self.history_leader_actions[idx]
        follower_actions = self.history_follower_actions[idx]
        
        for i in range(self.num_followers):
            phi_val = phi_cross(state, leader_action)
            u_pred = phi_val @ self.W[i]
            u_actual = follower_actions[i*2:(i+1)*2]
            error = u_actual - u_pred
            total_grad[i] += error[0] * phi_val  # 简化
    
    # 平均梯度更新
    for i in range(self.num_followers):
        self.W[i] += self.eta_W * total_grad[i] / batch_size
```

### 方向 5：更好的控制策略

**当前问题**：`compute_optimal_action` 使用简化的不动点迭代。

**改进方案**：
```python
def compute_optimal_action_mpc(self, state, horizon=10):
    """
    使用模型预测控制 (MPC) 计算最优动作。
    
    1. 用学习的模型预测未来 horizon 步
    2. 优化序列 u0[t:t+horizon] 最小化累积代价
    3. 只执行第一步 u0[t]
    """
    # TODO: 实现 MPC
    pass
```

### 方向 6：融合 Follower 的 W 学习

**当前问题**：Leader 和 Follower 独立学习，可能导致系统不稳定。

**改进方案**：
```python
# 在 Leader 中估计 follower 的 W_hat
class AGDLeader:
    def __init__(self, ...):
        # 估计的 follower W
        self.W_hat_followers = [np.zeros(5) for _ in range(num_followers)]
    
    def estimate_follower_W(self, state, leader_action, follower_actions):
        """使用观测数据估计 follower 的 W"""
        # TODO: 实现 W_hat 估计
        pass
    
    def predict_follower_action(self, state, leader_action, i):
        # 使用估计的 W_hat 而不是直接学习的 W
        grad_Vi = value_gradient(state, self.W_hat_followers[i])
        di = self.current_lambda[i, 0]  # 估计的 di
        return -di * leader_action - 0.5 * gi(state) * (gi(state).T @ grad_Vi)
```

---

## 诊断与可视化

### 关键指标

| 指标 | 含义 | 理想行为 |
|------|------|---------|
| 预测误差 | `||u_pred - u_actual||` | 下降并趋于稳定 |
| W 范数 | `||W||` | 有界，不发散 |
| 系统状态 | `||x||` | 收敛到原点 |
| Leader 代价 | `J0` | 下降并稳定 |

### 诊断图说明

1. **系统状态轨迹**：应看到状态收敛到原点附近
2. **Agent 动作**：不应过早归零（除非已收敛）
3. **预测误差**：应随学习下降
4. **W 范数**：应有界，持续增长表示不稳定
5. **Value 函数**：应随状态收敛而下降

---

## 使用示例

```python
from test_agd_leader_fixed import AGDLeader, run_training, plot_results

# 运行训练
env, leader, followers, history = run_training(max_iters=200, seed=42)

# 可视化
plot_results(env, leader, followers, history)

# 分析结果
print(f"最终预测误差：{np.mean(history['prediction_errors'][-10:]):.4f}")
print(f"最终代价：{np.mean(history['costs'][-10:]):.4f}")
```

---

## 总结

`AGDLeader` 提供了一个**完全基于观测**的领导者控制框架：

**优点**：
- ✓ 不需要 follower 内部参数
- ✓ 实现简单，易于调试
- ✓ 可迁移到类似环境

**局限**：
- ✗ 基函数选择影响性能
- ✗ 需要手动调整学习率
- ✗ 可能收敛到局部最优

**改进优先级**：
1. 正则化（防止过拟合）
2. 自适应学习率
3. 更好的基函数
4. 批量更新
5. MPC 控制
