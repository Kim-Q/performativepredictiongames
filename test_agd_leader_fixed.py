"""
基于自适应梯度下降的 Env2 环境领导者控制（修复版）
====================================================
修复问题:
1. Follower 的 W 权重无法正确学习
2. 50 步后 action 归零但状态仍在变化

解决方案:
1. 使用原始 exm2.py 中的 FollowerAgent（W 能正确学习）
2. Leader 通过观测学习 follower 响应模型
3. 增加诊断信息帮助理解系统行为
"""

import numpy as np
from exm2 import Env2, FollowerAgent, LeaderAgent, g0_vec, g1_vec, g2_vec, g3_vec, phi, value, value_gradient
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
rcParams['axes.unicode_minus'] = False


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
        x1, x2, u0, u1,
        x1*u0, x1*u1, x2*u0, x2*u1,
        x1**2, x2**2, u0**2, u1**2
    ])


def phi_cross_dim() -> int:
    return 12


# ==============================================================================
# 观测型领导者（不依赖 follower 内部信息）
# ==============================================================================

class AGDLeader:
    """
    基于观测的领导者 - 完全不知道 follower 的内部参数。
    通过观测 (state, follower_actions) 学习 follower 的响应模型。
    """
    
    def __init__(self, env: Env2, num_followers: int = 3, seed: int = 42):
        self.env = env
        self.num_followers = num_followers
        self.rng = np.random.default_rng(seed)
        
        self.phi_dim = phi_cross_dim()
        self.W = [np.zeros(self.phi_dim) for _ in range(num_followers)]
        self.K = np.zeros((2, 2 + num_followers * 2))
        
        self.history_states = []
        self.history_leader_actions = []
        self.history_follower_actions = []
        
        self.eta_W = 0.01
        self.eta_K = 0.001
        
        self.Q0 = 0.5 * np.eye(2)
        self.R0 = np.eye(2)
        self.C = [0.1, 0.1, 0.1]
        
        # 诊断信息
        self.W_norms = []
        
    def predict_follower_action(self, state: np.ndarray, leader_action: np.ndarray, i: int) -> np.ndarray:
        u_i_pred = phi_cross(state, leader_action) @ self.W[i]
        return u_i_pred
    
    def predict_all_followers(self, state: np.ndarray, leader_action: np.ndarray) -> List[np.ndarray]:
        return [self.predict_follower_action(state, leader_action, i) for i in range(self.num_followers)]
    
    def update_model(self, state: np.ndarray, leader_action: np.ndarray, 
                     actual_follower_actions: List[np.ndarray], eta: float = None):
        if eta is None:
            eta = self.eta_W
        
        for i in range(self.num_followers):
            phi_val = phi_cross(state, leader_action)
            u_pred = phi_val @ self.W[i]
            u_actual = actual_follower_actions[i]
            error = u_actual - u_pred
            
            for d in range(2):
                self.W[i] += eta * error[d] * phi_val
            
            self.W[i] = np.clip(self.W[i], -10, 10)
        
        # 记录 W 的范数用于诊断
        self.W_norms.append([np.linalg.norm(w) for w in self.W])
    
    def compute_optimal_action(self, state: np.ndarray) -> np.ndarray:
        u0_est = np.zeros(2)
        
        for _ in range(3):
            u_followers = self.predict_all_followers(state, u0_est)
            u_coupled = u0_est.copy()
            for i, ui in enumerate(u_followers):
                u_coupled += self.C[i] * ui
            u0_opt = -np.linalg.solve(self.R0, u_coupled - u0_est)
            u0_est = 0.7 * u0_est + 0.3 * u0_opt
        
        if len(self.history_states) < 50:
            u0_est += self.rng.uniform(-0.5, 0.5, size=2)
        
        return np.clip(u0_est, -5, 5)
    
    def step(self, state: np.ndarray) -> np.ndarray:
        self.history_states.append(state.copy())
        leader_action = self.compute_optimal_action(state)
        self.history_leader_actions.append(leader_action.copy())
        return leader_action
    
    def learn(self, state: np.ndarray, leader_action: np.ndarray, 
              follower_actions: List[np.ndarray]):
        self.history_follower_actions.append(np.concatenate(follower_actions).copy())
        self.update_model(state, leader_action, follower_actions)


# ==============================================================================
# 训练流程（使用原始 FollowerAgent）
# ==============================================================================

def run_training(max_iters: int = 200, seed: int = 42, use_original_follower: bool = True):
    """
    运行训练流程。
    
    参数:
        use_original_follower: 如果 True，使用 exm2.py 中的 FollowerAgent（W 能正确学习）
                              如果 False，使用简化的 follower（W 固定为零）
    """
    env = Env2()
    
    # 创建 follower
    followers = [
        FollowerAgent(env, qi=2.0, ri=1.0, g_func=g1_vec, index=1, seed=seed),
        FollowerAgent(env, qi=1.0, ri=1.0, g_func=g2_vec, index=2, seed=seed),
        FollowerAgent(env, qi=1.5, ri=1.0, g_func=g3_vec, index=3, seed=seed),
    ]
    
    # 创建 leader
    leader = AGDLeader(env, num_followers=3, seed=seed)
    
    history = {
        'states': [],
        'leader_actions': [],
        'follower_actions': [],
        'prediction_errors': [],
        'costs': [],
        'leader_values': [],
        'follower_values': [],
        'follower_W_norms': []  # 记录 follower 的 W 范数
    }
    
    print("="*60)
    print("AGD Leader 训练开始（修复版）")
    print("="*60)
    print(f"环境：Env2 (非线性系统)")
    print(f"Leader: 完全基于观测")
    print(f"Follower: {'原始 FollowerAgent (W 可学习)' if use_original_follower else '简化 Follower (W=0)'}")
    print(f"最大迭代次数：{max_iters}")
    print("="*60)
    
    for t in range(max_iters):
        state = env.state.copy()
        
        # Leader 行动
        leader_action = leader.step(state)
        
        # Follower 响应
        follower_actions = [f.compute_action(state, leader_action) for f in followers]
        
        # 关键修复：记录 rho 并更新 Follower 的 W
        for f in followers:
            f.record_rho(state, leader_action, follower_actions)
        
        # 更新 Follower 的 W（使用最小二乘法）
        for f in followers:
            f.update_W(env.chi, step_idx=t)
        
        # 清空缓冲区（与原始代码一致）
        for f in followers:
            f.Xi.clear()
        env.chi.clear()
        
        # Leader 学习
        leader.learn(state, leader_action, follower_actions)
        
        # 计算预测误差
        pred_errors = []
        for i in range(3):
            u_pred = leader.predict_follower_action(state, leader_action, i)
            u_actual = follower_actions[i]
            pred_errors.append(np.linalg.norm(u_pred - u_actual))
        
        # 计算代价
        u_coupled = leader_action.copy()
        for i, ui in enumerate(follower_actions):
            u_coupled += leader.C[i] * ui
        cost = state.T @ leader.Q0 @ state + u_coupled.T @ leader.R0 @ u_coupled
        
        # 计算 Values
        V_leader = cost
        V_followers = []
        for i, f in enumerate(followers):
            # 使用 follower 的真实 W 计算值函数
            Vi = value(state, f.W)
            V_followers.append(Vi)
        
        # 记录历史
        history['states'].append(state.copy())
        history['leader_actions'].append(leader_action.copy())
        history['follower_actions'].append(np.concatenate(follower_actions).copy())
        history['prediction_errors'].append(pred_errors)
        history['costs'].append(cost)
        history['leader_values'].append(V_leader)
        history['follower_values'].append(V_followers)
        history['follower_W_norms'].append([np.linalg.norm(f.W) for f in followers])
        
        # 环境步进
        env.step(leader_action, *[f.copy() for f in follower_actions])
        
        # 打印进度（包含诊断信息）
        if t % 20 == 0 or t == max_iters - 1:
            avg_pred_err = np.mean(pred_errors)
            avg_W_norm = np.mean([np.linalg.norm(f.W) for f in followers])
            print(f"迭代 {t}: 预测误差 = {avg_pred_err:.4f}, 代价 = {cost:.4f}, Follower W 范数 = {avg_W_norm:.4f}")
            print(f"  Leader 动作：{leader_action.round(4)}")
            print(f"  Follower 动作：{[u.round(2) for u in follower_actions]}")
            print(f"  系统状态：{state.round(4)}")
    
    print("\n训练完成。")
    return env, leader, followers, history


# ==============================================================================
# 绘图
# ==============================================================================

def plot_results(env, leader, followers, history):
    """绘制训练结果（包含诊断信息）。"""
    fig = plt.figure(figsize=(20, 14))
    
    states = np.array(history['states'])
    leader_actions = np.array(history['leader_actions'])
    follower_actions = np.array(history['follower_actions'])
    pred_errors = np.array(history['prediction_errors'])
    costs = np.array(history['costs'])
    leader_values = np.array(history['leader_values'])
    follower_values = np.array(history['follower_values'])
    follower_W_norms = np.array(history['follower_W_norms'])
    leader_W_norms = np.array(leader.W_norms)
    
    colors = ['blue', 'green', 'red']
    
    # 图 1: 系统状态
    ax = fig.add_subplot(4, 4, 1)
    ax.plot(states[:, 0], label='x1', color='blue', linewidth=2)
    ax.plot(states[:, 1], label='x2', color='red', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('状态值', fontsize=11)
    ax.set_title('系统状态轨迹', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 2: Leader 动作
    ax = fig.add_subplot(4, 4, 2)
    ax.plot(leader_actions[:, 0], label='u0[0]', color='blue', linewidth=2)
    ax.plot(leader_actions[:, 1], label='u0[1]', color='red', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('动作值', fontsize=11)
    ax.set_title('Leader 动作 (u0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 3: Follower 动作
    ax = fig.add_subplot(4, 4, 3)
    for i in range(3):
        ax.plot(follower_actions[:, i*2], label=f'u{i+1}[0]', color=colors[i], linestyle='-', linewidth=2)
        ax.plot(follower_actions[:, i*2+1], label=f'u{i+1}[1]', color=colors[i], linestyle='--', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('动作值', fontsize=11)
    ax.set_title('Follower 动作 (u1, u2, u3)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 4: 预测误差
    ax = fig.add_subplot(4, 4, 4)
    for i in range(3):
        ax.plot(pred_errors[:, i], label=f'Follower {i+1}', color=colors[i], linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('预测误差范数', fontsize=11)
    ax.set_title('Follower 响应模型预测误差', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 5: Leader 代价
    ax = fig.add_subplot(4, 4, 5)
    ax.plot(costs, color='purple', linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('代价 J0', fontsize=11)
    ax.set_title('Leader 代价函数', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图 6: 状态相图
    ax = fig.add_subplot(4, 4, 6)
    ax.plot(states[:, 0], states[:, 1], color='purple', alpha=0.5, linewidth=2)
    ax.plot(states[0, 0], states[0, 1], 'go', label='起点', markersize=8)
    ax.plot(states[-1, 0], states[-1, 1], 'rx', label='终点', markersize=8)
    ax.set_xlabel('x1', fontsize=11)
    ax.set_ylabel('x2', fontsize=11)
    ax.set_title('状态相图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 7: Leader Value
    ax = fig.add_subplot(4, 4, 7)
    ax.plot(leader_values, color='blue', linewidth=2, label='Leader Value')
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Leader 值函数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 8: Follower Values
    ax = fig.add_subplot(4, 4, 8)
    for i in range(3):
        ax.plot(follower_values[:, i], label=f'Follower {i+1} Value', color=colors[i], linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Follower 值函数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 9: Follower W 范数（诊断）
    ax = fig.add_subplot(4, 4, 9)
    for i in range(3):
        ax.plot(follower_W_norms[:, i], label=f'||W{i+1}||', color=colors[i], linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('W 范数', fontsize=11)
    ax.set_title('Follower 值函数权重范数（诊断）', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 10: Leader W 范数（诊断）
    ax = fig.add_subplot(4, 4, 10)
    for i in range(3):
        ax.plot(leader_W_norms[:, i], label=f'||W{i+1}||', color=colors[i], linewidth=2)
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('W 范数', fontsize=11)
    ax.set_title('Leader 响应模型权重范数（诊断）', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 11: Leader 动作相图
    ax = fig.add_subplot(4, 4, 11)
    ax.plot(leader_actions[:, 0], leader_actions[:, 1], color='blue', alpha=0.5, linewidth=2)
    ax.plot(leader_actions[0, 0], leader_actions[0, 1], 'go', label='起点', markersize=8)
    ax.plot(leader_actions[-1, 0], leader_actions[-1, 1], 'rx', label='终点', markersize=8)
    ax.set_xlabel('u0[0]', fontsize=11)
    ax.set_ylabel('u0[1]', fontsize=11)
    ax.set_title('Leader 动作相图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图 12: Follower 动作相图
    ax = fig.add_subplot(4, 4, 12)
    for i in range(3):
        ax.plot(follower_actions[:, i*2], follower_actions[:, i*2+1], 
               label=f'F{i+1}', color=colors[i], alpha=0.5, linewidth=2)
    ax.set_xlabel('ui[0]', fontsize=11)
    ax.set_ylabel('ui[1]', fontsize=11)
    ax.set_title('Follower 动作相图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('agd_leader_results_fixed.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到：agd_leader_results_fixed.png")


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("基于 AGD 的 Env2 环境领导者控制（修复版）")
    print("="*60)
    print("\n问题诊断:")
    print("  - 之前版本：50 步后 action 归零但状态仍在变化")
    print("  - 原因：Follower 的 W 权重无法正确学习")
    print("  - 修复：使用原始 FollowerAgent（W 可通过最小二乘法学习）")
    print("="*60 + "\n")
    
    env, leader, followers, history = run_training(max_iters=200, seed=42)
    plot_results(env, leader, followers, history)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\n关键结果:")
    print(f"  - 最终平均预测误差：{np.mean(history['prediction_errors'][-10:]):.4f}")
    print(f"  - 最终平均代价：{np.mean(history['costs'][-10:]):.4f}")
    print(f"  - 最终 Follower W 范数：{np.mean(history['follower_W_norms'][-10:], axis=0).round(4)}")
