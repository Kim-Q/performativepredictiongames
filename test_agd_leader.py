"""
基于自适应梯度下降的 Env2 环境领导者控制
==========================================
问题设定:
- Leader 完全不知道 Follower 的内部参数 (d, q, r 均未知)
- Leader 只能观测：state 和 follower 的 actions
- 目标：学习 follower 的响应模型，并优化自己的控制策略

方法:
1. 用基函数近似 follower 的响应策略：u_f = Φ(state, u0)^T @ W
2. 使用 AGD 在线估计 W
3. 基于估计的模型优化 leader 的控制 u0
"""

import numpy as np
from exm2 import Env2, FollowerAgent, g0_vec, g1_vec, g2_vec, g3_vec, phi, value, value_gradient
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==============================================================================
# 配置 matplotlib 以支持中文显示
# ==============================================================================
import platform
system = platform.system()
if system == 'Darwin':  # macOS
    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
elif system == 'Windows':
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
else:  # Linux
    rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ==============================================================================
# 基函数设计
# ==============================================================================

def phi_cross(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    交叉基函数：包含 state 和 action 的交叉项。
    
    Φ = [x1, x2, u[0], u[1], x1*u[0], x1*u[1], x2*u[0], x2*u[1], 
         x1^2, x2^2, u[0]^2, u[1]^2]
    
    共 12 维
    """
    x1, x2 = state
    u0, u1 = action
    return np.array([
        x1, x2,           # 状态线性项
        u0, u1,           # 动作线性项
        x1*u0, x1*u1,     # 状态 - 动作交叉项
        x2*u0, x2*u1,
        x1**2, x2**2,     # 状态二次项
        u0**2, u1**2      # 动作二次项
    ])


def phi_cross_dim() -> int:
    """返回交叉基函数的维度。"""
    return 12


# ==============================================================================
# 领导者类 - 完全基于观测
# ==============================================================================

class AGDLeader:
    """
    基于自适应梯度下降的领导者。
    
    关键特性:
    - 完全不知道 follower 的内部参数
    - 通过观测 (state, follower_actions) 学习 follower 的响应模型
    - 优化自己的控制策略以最小化系统代价
    
    参数估计:
    - 对每个 follower i，学习响应模型：u_i = Φ(state, u0)^T @ W_i
    - 使用 AGD 在线更新 W_i
    """
    
    def __init__(self, env: Env2, num_followers: int = 3, seed: int = 42):
        self.env = env
        self.num_followers = num_followers
        self.rng = np.random.default_rng(seed)
        
        # 基函数维度
        self.phi_dim = phi_cross_dim()
        
        # 每个 follower 的响应模型权重 (需要估计)
        # W[i] 用于预测 follower i 的动作
        self.W = [np.zeros(self.phi_dim) for _ in range(num_followers)]
        
        # 领导者的控制权重 (需要优化)
        # 用于计算最优控制 u0 = -K @ [state; 预测的 follower 动作]
        self.K = np.zeros((2, 2 + num_followers * 2))  # [state(2), u1(2), u2(2), u3(2)]
        
        # 历史数据用于学习
        self.history_states = []
        self.history_leader_actions = []
        self.history_follower_actions = []
        
        # 学习率
        self.eta_W = 0.01    # 模型估计学习率
        self.eta_K = 0.001   # 控制优化学习率
        
        # 代价函数参数 (leader 自己的，已知)
        self.Q0 = 0.5 * np.eye(2)
        self.R0 = np.eye(2)
        self.C = [0.1, 0.1, 0.1]  # 对 follower 动作的耦合权重
        
        # Value 函数权重（用于计算 agent values）
        self.V_leader = []
        self.V_followers = [[] for _ in range(num_followers)]
        
    def predict_follower_action(self, state: np.ndarray, leader_action: np.ndarray, 
                                 i: int) -> np.ndarray:
        """
        预测 follower i 的动作。
        
        u_i_pred = Φ(state, u0)^T @ W[i]
        """
        phi_val = phi_cross(state, leader_action)
        return phi_val @ self.W[i]
    
    def predict_all_followers(self, state: np.ndarray, leader_action: np.ndarray) -> List[np.ndarray]:
        """预测所有 follower 的动作。"""
        return [self.predict_follower_action(state, leader_action, i) 
                for i in range(self.num_followers)]
    
    def update_model(self, state: np.ndarray, leader_action: np.ndarray, 
                     actual_follower_actions: List[np.ndarray], eta: float = None):
        """
        使用 AGD 更新 follower 响应模型。
        
        对每个 follower i:
        - 计算预测误差：e_i = u_i_actual - u_i_pred
        - 梯度下降更新：W[i] += eta * e_i * Φ(state, u0)
        """
        if eta is None:
            eta = self.eta_W
        
        for i in range(self.num_followers):
            phi_val = phi_cross(state, leader_action)
            u_pred = phi_val @ self.W[i]
            u_actual = actual_follower_actions[i]
            
            # 预测误差
            error = u_actual - u_pred
            
            # SGD 更新 (对每个动作分量)
            for d in range(2):
                self.W[i] += eta * error[d] * phi_val
            
            # 稳定性：裁剪权重
            self.W[i] = np.clip(self.W[i], -10, 10)
    
    def compute_optimal_action(self, state: np.ndarray) -> np.ndarray:
        """
        基于估计的 follower 模型计算最优控制。
        
        Leader 的代价函数:
        J0 = x^T Q0 x + (u0 + Σ ci * ui)^T R0 (u0 + Σ ci * ui)
        
        使用估计的模型 ui = Φ(state, u0)^T @ W[i]，这是一个不动点问题。
        简化：使用上一步的预测作为 ui 的近似。
        """
        # 预测 follower 动作（使用当前 u0 的估计，初始为 0）
        u0_est = np.zeros(2)
        
        for _ in range(3):  # 不动点迭代
            u_followers = self.predict_all_followers(state, u0_est)
            
            # 计算耦合项：u_coupled = u0 + Σ ci * ui
            u_coupled = u0_est.copy()
            for i, ui in enumerate(u_followers):
                u_coupled += self.C[i] * ui
            
            # 最优控制（忽略 u0 对 ui 的影响，简化）
            # u0* = -R0^{-1} * (Σ ci * ui)
            u0_opt = -np.linalg.solve(self.R0, u_coupled - u0_est)
            
            # 平滑更新
            u0_est = 0.7 * u0_est + 0.3 * u0_opt
        
        # 添加探索噪声
        if len(self.history_states) < 50:
            u0_est += self.rng.uniform(-0.5, 0.5, size=2)
        
        return np.clip(u0_est, -5, 5)
    
    def update_control_gain(self, state: np.ndarray, leader_action: np.ndarray,
                            follower_actions: List[np.ndarray], eta: float = None):
        """
        更新控制增益 K。
        
        基于代价梯度：∇J0 = 2 * Q0 @ x + 2 * R0 @ u_coupled
        """
        if eta is None:
            eta = self.eta_K
        
        # 计算耦合动作
        u_coupled = leader_action.copy()
        for i, ui in enumerate(follower_actions):
            u_coupled += self.C[i] * ui
        
        # 代价梯度
        grad_state = 2 * self.Q0 @ state
        grad_action = 2 * self.R0 @ u_coupled
        
        # 更新 K（简化：直接梯度下降）
        # K 的输入是 [state; u1; u2; u3]
        input_vec = np.concatenate([state] + follower_actions)
        
        # 梯度：dJ/dK = grad_action @ input^T
        grad_K = np.outer(grad_action, input_vec)
        
        self.K -= eta * grad_K
        self.K = np.clip(self.K, -10, 10)
    
    def step(self, state: np.ndarray) -> np.ndarray:
        """
        执行一步：观测 → 预测 → 优化 → 行动。
        
        返回 leader 的动作。
        """
        # 记录历史
        self.history_states.append(state.copy())
        
        # 计算最优动作
        leader_action = self.compute_optimal_action(state)
        self.history_leader_actions.append(leader_action.copy())
        
        return leader_action
    
    def learn(self, state: np.ndarray, leader_action: np.ndarray, 
              follower_actions: List[np.ndarray]):
        """
        学习步骤：更新模型和控制增益。
        """
        self.history_follower_actions.append(np.concatenate(follower_actions).copy())
        
        # 更新 follower 响应模型
        self.update_model(state, leader_action, follower_actions)
        
        # 更新控制增益
        self.update_control_gain(state, leader_action, follower_actions)


# ==============================================================================
# 训练流程
# ==============================================================================

def run_training(max_iters: int = 200, seed: int = 42):
    """
    运行训练流程。
    """
    # 创建环境
    env = Env2()
    
    # 创建 follower（对 leader 完全隐藏）
    followers = [
        FollowerAgent(env, qi=2.0, ri=1.0, g_func=g1_vec, index=1, seed=seed),
        FollowerAgent(env, qi=1.0, ri=1.0, g_func=g2_vec, index=2, seed=seed),
        FollowerAgent(env, qi=1.5, ri=1.0, g_func=g3_vec, index=3, seed=seed),
    ]
    
    # 创建 leader（不知道 follower 的任何内部信息）
    leader = AGDLeader(env, num_followers=3, seed=seed)
    
    # 历史记录
    history = {
        'states': [],
        'leader_actions': [],
        'follower_actions': [],
        'prediction_errors': [],
        'costs': [],
        'leader_values': [],
        'follower_values': []
    }
    
    print("="*60)
    print("AGD Leader 训练开始")
    print("="*60)
    print(f"环境：Env2 (非线性系统)")
    print(f"Leader: 完全基于观测，不知道 follower 内部参数")
    print(f"最大迭代次数：{max_iters}")
    print("="*60)
    
    for t in range(max_iters):
        state = env.state.copy()
        
        # Leader 行动
        leader_action = leader.step(state)
        
        # Follower 响应（leader 只能观测结果）
        follower_actions = [f.compute_action(state, leader_action) for f in followers]
        
        # Leader 学习
        leader.learn(state, leader_action, follower_actions)
        
        # 计算预测误差（用于评估）
        pred_errors = []
        for i in range(3):
            u_pred = leader.predict_follower_action(state, leader_action, i)
            u_actual = follower_actions[i]
            pred_errors.append(np.linalg.norm(u_pred - u_actual))
        
        # 计算 leader 代价（用于评估）
        u_coupled = leader_action.copy()
        for i, ui in enumerate(follower_actions):
            u_coupled += leader.C[i] * ui
        cost = state.T @ leader.Q0 @ state + u_coupled.T @ leader.R0 @ u_coupled
        
        # 计算 Agent Values（值函数）
        # Leader value: V0 = state^T Q0 state + u_coupled^T R0 u_coupled
        V_leader = cost
        
        # Follower values: Vi = qi * ||state||^2 + ri * ||ui + di * u0||^2
        # 由于 leader 不知道 qi, ri, di，使用估计的响应模型近似
        V_followers = []
        for i in range(3):
            # 使用观测到的实际动作计算近似值函数
            ui = follower_actions[i]
            # Vi ≈ ||state||^2 + ||ui||^2 (简化近似)
            Vi = 0.5 * state.T @ state + 0.5 * ui.T @ ui
            V_followers.append(Vi)
        
        # 记录历史
        history['states'].append(state.copy())
        history['leader_actions'].append(leader_action.copy())
        history['follower_actions'].append(np.concatenate(follower_actions).copy())
        history['prediction_errors'].append(pred_errors)
        history['costs'].append(cost)
        history['leader_values'].append(V_leader)
        history['follower_values'].append(V_followers)
        
        # 环境步进
        env.step(leader_action, *[f.copy() for f in follower_actions])
        
        # 打印进度
        if t % 20 == 0 or t == max_iters - 1:
            avg_pred_err = np.mean(pred_errors)
            print(f"迭代 {t}: 平均预测误差 = {avg_pred_err:.4f}, 代价 = {cost:.4f}")
            print(f"  Leader 动作：{leader_action.round(4)}")
            print(f"  Follower 动作：{[u.round(2) for u in follower_actions]}")
    
    print("\n训练完成。")
    return env, leader, followers, history


# ==============================================================================
# 绘图
# ==============================================================================

def plot_results(env, leader, history):
    """绘制训练结果。"""
    fig = plt.figure(figsize=(18, 12))
    
    states = np.array(history['states'])
    leader_actions = np.array(history['leader_actions'])
    follower_actions = np.array(history['follower_actions'])
    pred_errors = np.array(history['prediction_errors'])
    costs = np.array(history['costs'])
    leader_values = np.array(history['leader_values'])
    follower_values = np.array(history['follower_values'])
    
    colors = ['blue', 'green', 'red']
    
    # 图 1: 系统状态
    ax = fig.add_subplot(3, 4, 1)
    ax.plot(states[:, 0], label='x1', color='blue', linewidth=2)
    ax.plot(states[:, 1], label='x2', color='red', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('状态值', fontsize=11)
    ax.set_title('系统状态轨迹', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 2: Leader 动作
    ax = fig.add_subplot(3, 4, 2)
    ax.plot(leader_actions[:, 0], label='u0[0]', color='blue', linewidth=2)
    ax.plot(leader_actions[:, 1], label='u0[1]', color='red', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('动作值', fontsize=11)
    ax.set_title('Leader 动作 (u0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 3: Follower 动作
    ax = fig.add_subplot(3, 4, 3)
    for i in range(3):
        ax.plot(follower_actions[:, i*2], label=f'u{i+1}[0]', color=colors[i], linestyle='-', linewidth=2)
        ax.plot(follower_actions[:, i*2+1], label=f'u{i+1}[1]', color=colors[i], linestyle='--', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('动作值', fontsize=11)
    ax.set_title('Follower 动作 (u1, u2, u3)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 4: 预测误差
    ax = fig.add_subplot(3, 4, 4)
    for i in range(3):
        ax.plot(pred_errors[:, i], label=f'Follower {i+1}', color=colors[i], linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('预测误差范数', fontsize=11)
    ax.set_title('Follower 响应模型预测误差', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 5: Leader 代价
    ax = fig.add_subplot(3, 4, 5)
    ax.plot(costs, color='purple', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('代价 J0', fontsize=11)
    ax.set_title('Leader 代价函数', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图 6: 状态相图
    ax = fig.add_subplot(3, 4, 6)
    ax.plot(states[:, 0], states[:, 1], color='purple', alpha=0.5, linewidth=2)
    ax.plot(states[0, 0], states[0, 1], 'go', label='起点', markersize=8)
    ax.plot(states[-1, 0], states[-1, 1], 'rx', label='终点', markersize=8)
    ax.set_xlabel('x1', fontsize=11)
    ax.set_ylabel('x2', fontsize=11)
    ax.set_title('状态相图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 7: Leader Value
    ax = fig.add_subplot(3, 4, 7)
    ax.plot(leader_values, color='blue', linewidth=2, label='Leader Value')
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Leader 值函数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 8: Follower Values
    ax = fig.add_subplot(3, 4, 8)
    for i in range(3):
        ax.plot(follower_values[:, i], label=f'Follower {i+1} Value', color=colors[i], linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Follower 值函数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 9-11: 模型权重
    for i in range(3):
        ax = fig.add_subplot(3, 4, 9+i)
        W = leader.W[i]
        ax.bar(range(len(W)), W, color=colors[i], alpha=0.7)
        ax.set_xlabel('基函数索引', fontsize=10)
        ax.set_ylabel('权重值', fontsize=10)
        ax.set_title(f'Follower {i+1} 响应模型权重', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 图 12: Leader 动作相图
    ax = fig.add_subplot(3, 4, 12)
    ax.plot(leader_actions[:, 0], leader_actions[:, 1], color='blue', alpha=0.5, linewidth=2)
    ax.plot(leader_actions[0, 0], leader_actions[0, 1], 'go', label='起点', markersize=8)
    ax.plot(leader_actions[-1, 0], leader_actions[-1, 1], 'rx', label='终点', markersize=8)
    ax.set_xlabel('u0[0]', fontsize=11)
    ax.set_ylabel('u0[1]', fontsize=11)
    ax.set_title('Leader 动作相图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('agd_leader_results.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到：agd_leader_results.png")


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("基于 AGD 的 Env2 环境领导者控制")
    print("="*60)
    print("\n问题设定:")
    print("  - Leader 完全不知道 Follower 的内部参数 (d, q, r)")
    print("  - Leader 只能观测 state 和 follower actions")
    print("  - 目标：学习 follower 响应模型并优化控制")
    print("="*60 + "\n")
    
    env, leader, followers, history = run_training(max_iters=200, seed=42)
    plot_results(env, leader, history)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\n关键结果:")
    print(f"  - 最终平均预测误差：{np.mean(history['prediction_errors'][-10:]):.4f}")
    print(f"  - 最终平均代价：{np.mean(history['costs'][-10:]):.4f}")
    print("\n方法特点:")
    print("  ✓ 完全基于观测，不需要 follower 内部信息")
    print("  ✓ 在线学习 follower 响应模型")
    print("  ✓ 自适应优化控制策略")
    print("\n进一步改进:")
    print("  - 使用更丰富的基函数（如 RBF 核）")
    print("  - 添加正则化防止过拟合")
    print("  - 使用更复杂的控制策略（如 MPC）")
