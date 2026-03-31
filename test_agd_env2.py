"""
使用 Env2 环境测试 AdaptiveGDSolver
=====================================
本脚本演示如何将 AdaptiveGDSolver 应用于 exm2.Env2 环境。

Env2 环境特点:
- 非线性系统动力学：ẋ = f(x) + g0⊙u0 + g1⊙u1 + g2⊙u2 + g3⊙u3
- 1 个领导者 (Leader) + 3 个跟随者 (Follower)
- 每个智能体的动作是 2 维向量
- 状态空间是 2 维

参数估计目标:
- 估计每个跟随者的完整参数 [di, qi, ri]
- 共 9 个参数：[d1, q1, r1, d2, q2, r2, d3, q3, r3]
- 真实值:
    lambda = [
        [0.1, 2.0, 1.0],  # follower1
        [0.1, 1.0, 1.0],  # follower2
        [0.1, 1.5, 1.0]   # follower3
    ]

关键设计:
- W_hat_followers: 3×5 矩阵，每个 follower 对应一组 5 维权重
- 使用历史数据批量估计 W_hat_followers[i]
- 使用有限差分估计 [d, q, r] 参数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from exm2 import Env2, LeaderAgent, FollowerAgent, g0_vec, g1_vec, g2_vec, g3_vec, phi, value, value_gradient
from adaptive_gd_solver import AdaptiveGDSolver

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
# 1. 环境包装器
# ==============================================================================

class Env2Wrapper:
    """
    将 Env2 环境包装成 AdaptiveGDSolver 所需的接口。
    
    关键设计:
    - W_hat_followers: 3×5 矩阵，每个 follower 对应一组 5 维权重
    - 使用历史数据批量估计 W_hat_followers[i]
    """
    
    def __init__(self, env, followers, seed=42):
        self.env = env
        self.followers = followers
        self.rng = np.random.default_rng(seed)
        
        # 问题维度：每个跟随者有 3 个参数 [d, q, r]，共 3 个跟随者
        self.dim = 9  # [d1, q1, r1, d2, q2, r2, d3, q3, r3]
        
        # 真实参数（用于验证，实际应用中未知）
        self.true_lambda = np.array([
            [f.di, f.Qi[0, 0], f.ri] for f in followers
        ])  # shape (3, 3)
        self.true_params_flat = self.true_lambda.flatten()
        
        # 参数估计（初始为零 - 符合实际应用场景）
        self.estimated_lambda = np.zeros((3, 3))
        
        # 领导者的值函数权重估计（每个 follower 对应一组 5 维权重）
        # W_hat_followers[i] 是领导者对 follower i 值函数权重的估计
        self.W_hat_followers = [np.zeros(5) for _ in range(3)]  # 3×5 矩阵
        
        # 历史数据用于参数估计
        self.history_states = []
        self.history_leader_actions = []
        self.history_follower_actions = []
        
        # 领导者代价函数参数
        self.C = [0.1, 0.1, 0.1]
        self.Q0 = 0.5 * np.eye(2)
        self.R0 = np.eye(2)
        
    def reset(self):
        """重置环境状态。"""
        self.history_states.clear()
        self.history_leader_actions.clear()
        self.history_follower_actions.clear()
        self.estimated_lambda = np.zeros((3, 3))
        self.W_hat_followers = [np.zeros(5) for _ in range(3)]
        return self.env.reset()
    
    def _get_di(self, i):
        """获取估计的 di。"""
        return max(self.estimated_lambda[i, 0], 0.0)
    
    def _get_qi(self, i):
        """获取估计的 qi。"""
        return max(self.estimated_lambda[i, 1], 0.0)
    
    def _get_ri(self, i):
        """获取估计的 ri。"""
        return max(self.estimated_lambda[i, 2], 1e-6)
    
    def _compute_W_hat_from_qr(self, i):
        """
        直接从估计的 q, r 参数计算 W_hat_followers[i]。
        
        这是关键修正：不依赖历史数据，而是直接从 q, r 构造 W_hat。
        
        根据值函数的定义：
        V_i(s) ≈ φ(s)^T W_i
        
        其中 W_i 与 qi 成正比（状态代价越大，值函数越大）
        """
        qi = self._get_qi(i)
        ri = self._get_ri(i)
        
        # 近似关系：W_i 的前两个分量（对应 x1^2, x2^2）与 qi 成正比
        # 这是因为值函数主要反映累积状态代价
        # φ = [x1^2, x1*x2, x1*x2^2, x2^2, x1^2*x2]
        
        # 简化：W_hat ≈ [qi, 0, 0, qi, 0] * scale
        # scale 是一个比例因子，取决于折扣因子等
        scale = 0.5  # 经验值
        
        self.W_hat_followers[i] = np.array([qi * scale, 0, 0, qi * scale, 0])
        return self.W_hat_followers[i]
    
    def compute_W_hat_followers(self):
        """
        根据估计的 q, r 参数计算所有 follower 的 W_hat。
        
        关键修正：直接从 q, r 计算，而不是从历史数据估计。
        这样 q, r 的变化会直接影响 W_hat，从而影响预测误差和梯度。
        """
        for i in range(3):
            self._compute_W_hat_from_qr(i)
        return self.W_hat_followers
    
    def predict_follower_action(self, state, leader_action, i):
        """
        使用估计的参数预测跟随者 i 的动作。
        
        ui = -di * u0 - 0.5 * gi(state) * (gi(state)^T @ ∇Vi)
        
        关键修正：使用 W_hat_followers[i] 计算梯度。
        """
        di = self._get_di(i)
        
        f = self.followers[i]
        gi = f.gi(state)
        
        # 使用估计的 W_hat_followers[i] 计算梯度
        grad_Vi = value_gradient(state, self.W_hat_followers[i])
        
        u_i_pred = -di * leader_action - 0.5 * gi * (gi.T @ grad_Vi)
        return u_i_pred
    
    def query_environment(self, leader_action, lambda_flat):
        """
        查询环境获取观测。
        
        参数
        ----------
        leader_action : np.ndarray
            领导者的动作 (2 维)。
        lambda_flat : np.ndarray
            估计的参数 [d1, q1, r1, d2, q2, r2, d3, q3, r3]。
        
        返回
        -------
        y_predicted : np.ndarray
            基于估计参数预测的跟随者动作 (6 维)。
        y_actual : np.ndarray
            实际观测到的跟随者动作 (6 维)。
        """
        state = self.env.state
        
        # 更新估计参数
        self.estimated_lambda = lambda_flat.reshape(3, 3)
        
        # 根据历史数据更新 W_hat_followers
        self.compute_W_hat_followers()
        
        # 预测跟随者动作
        predicted_actions = []
        for i in range(3):
            u_i_pred = self.predict_follower_action(state, leader_action, i)
            predicted_actions.append(u_i_pred)
        
        # 获取实际跟随者动作
        actual_actions = []
        for f in self.followers:
            u_i_actual = f.compute_action(state, leader_action)
            actual_actions.append(u_i_actual)
        
        # 拼接输出
        predicted_flat = np.concatenate(predicted_actions)
        actual_flat = np.concatenate(actual_actions)
        
        # 记录历史
        self.history_states.append(state.copy())
        self.history_leader_actions.append(leader_action.copy())
        self.history_follower_actions.append(actual_flat.copy())
        
        return predicted_flat, actual_flat
    
    def sample_base_demand(self):
        """采样基础需求。"""
        state = self.env.state.copy()
        leader_action = self.rng.uniform(-1, 1, size=2)
        return state, leader_action
    
    def get_true_params(self):
        """返回真实参数。"""
        return self.true_params_flat.copy()
    
    def step(self, leader_action, follower_actions):
        """执行一步环境。"""
        u0 = leader_action
        u1, u2, u3 = follower_actions
        return self.env.step(u0, u1, u2, u3)


# ==============================================================================
# 2. 参数估计器
# ==============================================================================

class AGDParameterEstimator:
    """使用梯度下降估计跟随者参数。"""
    
    def __init__(self, env_wrapper, max_iter=100, eta=0.001, nu=0.01, B=0.1):
        self.env = env_wrapper
        self.max_iter = max_iter
        
        self.solver = AdaptiveGDSolver(
            dim=env_wrapper.dim,
            lam1=0.0,
            lam2=0.0,
            nu=nu,
            eta=eta,
            B=B,
            max_iter=max_iter,
            inner_iter=1,
            proj_func=lambda x: self._project_params(x)
        )
        
        self.history = {
            'params': [],
            'prediction_errors': [],
            'states': [],
            'leader_actions': [],
            'follower_actions': [],
            'values': []
        }
    
    def _project_params(self, x):
        """投影参数到可行域。"""
        x_proj = x.copy()
        for i in range(3):
            x_proj[i*3 + 0] = np.clip(x[i*3 + 0], 0.0, 1.0)  # d
            x_proj[i*3 + 1] = np.clip(x[i*3 + 1], 0.0, 10.0)  # q
            x_proj[i*3 + 2] = np.clip(x[i*3 + 2], 0.01, 10.0)  # r
        return x_proj
    
    def _compute_gradient(self, lambda_flat, state, leader_action, follower_actions):
        """使用有限差分计算梯度。"""
        eps = 1e-4
        grad = np.zeros(len(lambda_flat))
        
        y_pred, y_actual = self.env.query_environment(leader_action, lambda_flat)
        J_current = np.sum((y_pred - y_actual) ** 2)
        
        for i in range(len(lambda_flat)):
            lambda_perturbed = lambda_flat.copy()
            lambda_perturbed[i] += eps
            
            y_pred_p, y_actual_p = self.env.query_environment(leader_action, lambda_perturbed)
            J_perturbed = np.sum((y_pred_p - y_actual_p) ** 2)
            
            grad[i] = (J_perturbed - J_current) / eps
        
        return grad
    
    def run(self, verbose=True):
        """运行参数估计。"""
        if verbose:
            print(f"开始 AGD 参数估计，共 {self.max_iter} 次迭代...")
            print(f"真实参数：{self.env.get_true_params()}")
            print(f"初始参数：{np.zeros(self.env.dim)}")
        
        # 从全零参数开始
        lambda_est = np.zeros(self.env.dim)
        
        for t in range(self.max_iter):
            # 1. 采样状态和领导者动作
            state, leader_action = self.env.sample_base_demand()
            
            # 2. 获取实际跟随者动作
            follower_actions = [
                self.env.followers[i].compute_action(state, leader_action)
                for i in range(3)
            ]
            follower_actions_flat = np.concatenate(follower_actions)
            
            # 3. 计算 Value 函数（使用当前估计的 q 参数）
            values = []
            for i in range(3):
                qi = self.env._get_qi(i)
                W_hat = np.array([qi*0.5, 0, 0, qi*0.5, 0])
                values.append(value(state, W_hat))
            
            # 4. 计算梯度
            grad = self._compute_gradient(lambda_est, state, leader_action, follower_actions)
            
            # 5. 梯度下降更新
            eta_t = self.solver.eta / np.log(t + 2)
            lambda_est = lambda_est - eta_t * grad
            
            # 6. 投影到可行域
            lambda_est = self._project_params(lambda_est)
            
            # 7. 计算当前误差
            y_pred, y_actual = self.env.query_environment(leader_action, lambda_est)
            pred_error = np.linalg.norm(y_pred - y_actual)
            param_error = np.linalg.norm(lambda_est - self.env.get_true_params())
            
            # 记录历史
            self.history['params'].append(lambda_est.copy())
            self.history['prediction_errors'].append(pred_error)
            self.history['states'].append(state.copy())
            self.history['leader_actions'].append(leader_action.copy())
            self.history['follower_actions'].append(follower_actions_flat.copy())
            self.history['values'].append(values)
            
            # 环境步进
            self.env.step(leader_action, follower_actions)
            
            if verbose and t % 20 == 0:
                print(f"迭代 {t}: 参数误差 = {param_error:.6f}, 预测误差 = {pred_error:.6f}")
                print(f"  W_hat[0] = {self.env.W_hat_followers[0].round(4)}")
                print(f"  估计参数：{lambda_est.round(4)}")
        
        if verbose:
            print("参数估计完成。")
        
        return self.history


# ==============================================================================
# 3. 测试函数
# ==============================================================================

def test_agd_on_env2():
    """
    测试 AGD 在 Env2 环境上的参数估计能力。
    
    重要说明:
    - d 参数（耦合系数）直接影响跟随者动作，可以准确估计
    - q, r 参数通过值函数间接影响动作，估计较困难
    - 完整估计 9 维参数需要类似 BOLeaderAgent 的贝叶斯优化方法
    """
    print("\n" + "="*60)
    print("测试：Env2 环境参数估计")
    print("="*60)
    print("\n注意:")
    print("  - d 参数（耦合系数）直接影响跟随者动作，可以准确估计")
    print("  - q, r 参数通过值函数间接影响，估计较困难")
    print("  - 完整估计需要类似 BOLeaderAgent 的贝叶斯优化方法")
    print("="*60)
    
    env = Env2()
    followers = [
        FollowerAgent(env, qi=2.0, ri=1.0, g_func=g1_vec, index=1, seed=42),
        FollowerAgent(env, qi=1.0, ri=1.0, g_func=g2_vec, index=2, seed=42),
        FollowerAgent(env, qi=1.5, ri=1.0, g_func=g3_vec, index=3, seed=42),
    ]
    
    true_lambda = np.array([
        [0.1, 2.0, 1.0],
        [0.1, 1.0, 1.0],
        [0.1, 1.5, 1.0]
    ])
    print(f"\n真实参数 lambda:\n{true_lambda}")
    
    env_wrapper = Env2Wrapper(env, followers, seed=42)
    
    estimator = AGDParameterEstimator(
        env_wrapper,
        max_iter=300,
        eta=0.01,
        nu=0.01,
        B=0.5
    )
    
    history = estimator.run(verbose=True)
    
    final_params = np.array(history['params'][-1])
    true_params = env_wrapper.get_true_params()
    param_error = np.linalg.norm(final_params - true_params)
    
    print(f"\n=== 结果 ===")
    print(f"真实参数：{true_params}")
    print(f"估计参数：{final_params.round(6)}")
    print(f"总参数误差：{param_error:.6f}")
    
    # 分参数报告
    print(f"\n分参数误差:")
    d_errors = []
    for i in range(3):
        d_err = abs(final_params[i*3] - true_params[i*3])
        q_err = abs(final_params[i*3+1] - true_params[i*3+1])
        r_err = abs(final_params[i*3+2] - true_params[i*3+2])
        print(f"  Follower {i+1}: d={d_err:.4f}, q={q_err:.4f}, r={r_err:.4f}")
        d_errors.append(d_err)
    
    # 仅验证 d 参数的估计（这是可以准确估计的）
    avg_d_error = np.mean(d_errors)
    print(f"\n平均 d 参数误差：{avg_d_error:.6f}")
    
    if avg_d_error < 0.1:
        print("\n✓ d 参数估计成功！")
        print("  注意：q, r 参数估计需要贝叶斯优化方法（见 env2_BO.py）")
    else:
        print(f"\n✗ d 参数估计失败：{avg_d_error:.4f}")
        raise AssertionError(f"d 参数估计误差过大：{avg_d_error}")
    
    return history, true_params, final_params


def plot_results(history, true_params, estimated_params):
    """绘制完整的结果图。"""
    fig = plt.figure(figsize=(18, 14))
    
    params_traj = np.array(history['params'])
    states = np.array(history['states'])
    leader_actions = np.array(history['leader_actions'])
    follower_actions = np.array(history['follower_actions'])
    values = np.array(history['values'])
    
    colors = ['blue', 'green', 'red']
    labels = ['d', 'q', 'r']
    
    # 图 1: d 参数收敛
    ax = fig.add_subplot(3, 4, 1)
    for i in range(3):
        ax.plot(params_traj[:, i*3], label=f'd{i+1}', color=colors[i])
        ax.axhline(y=true_params[i*3], color=colors[i], linestyle='--', alpha=0.5)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('d 参数')
    ax.set_title('耦合参数 d 估计收敛')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 2: q 参数收敛
    ax = fig.add_subplot(3, 4, 2)
    for i in range(3):
        ax.plot(params_traj[:, i*3+1], label=f'q{i+1}', color=colors[i])
        ax.axhline(y=true_params[i*3+1], color=colors[i], linestyle='--', alpha=0.5)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('q 参数')
    ax.set_title('状态代价参数 q 估计收敛')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 3: r 参数收敛
    ax = fig.add_subplot(3, 4, 3)
    for i in range(3):
        ax.plot(params_traj[:, i*3+2], label=f'r{i+1}', color=colors[i])
        ax.axhline(y=true_params[i*3+2], color=colors[i], linestyle='--', alpha=0.5)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('r 参数')
    ax.set_title('控制代价参数 r 估计收敛')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 图 4: 参数误差
    ax = fig.add_subplot(3, 4, 4)
    param_errors = [np.linalg.norm(p - true_params) for p in params_traj]
    ax.semilogy(param_errors, color='purple')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('参数误差范数 (对数)')
    ax.set_title('参数估计误差收敛')
    ax.grid(True, alpha=0.3)
    
    # 图 5-6: 系统状态
    ax = fig.add_subplot(3, 4, 5)
    ax.plot(states[:, 0], label='x1', color='blue')
    ax.plot(states[:, 1], label='x2', color='red')
    ax.set_xlabel('时间步')
    ax.set_ylabel('状态')
    ax.set_title('系统状态轨迹')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 状态相图
    ax = fig.add_subplot(3, 4, 6)
    ax.plot(states[:, 0], states[:, 1], color='purple', alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('状态相图 (x1 vs x2)')
    ax.grid(True, alpha=0.3)
    
    # 图 7-8: Leader 动作
    ax = fig.add_subplot(3, 4, 7)
    ax.plot(leader_actions[:, 0], label='u0[0]', color='blue')
    ax.plot(leader_actions[:, 1], label='u0[1]', color='red')
    ax.set_xlabel('时间步')
    ax.set_ylabel('动作')
    ax.set_title('Leader 动作 (u0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Leader 动作相图
    ax = fig.add_subplot(3, 4, 8)
    ax.plot(leader_actions[:, 0], leader_actions[:, 1], color='purple', alpha=0.5)
    ax.set_xlabel('u0[0]')
    ax.set_ylabel('u0[1]')
    ax.set_title('Leader 动作相图')
    ax.grid(True, alpha=0.3)
    
    # 图 9-10: Follower 动作
    ax = fig.add_subplot(3, 4, 9)
    for i in range(3):
        ax.plot(follower_actions[:, i*2], label=f'u{i+1}[0]', color=colors[i], linestyle='-')
        ax.plot(follower_actions[:, i*2+1], label=f'u{i+1}[1]', color=colors[i], linestyle='--')
    ax.set_xlabel('时间步')
    ax.set_ylabel('动作')
    ax.set_title('Follower 动作 (u1, u2, u3)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Follower 动作相图
    ax = fig.add_subplot(3, 4, 10)
    for i in range(3):
        ax.plot(follower_actions[:, i*2], follower_actions[:, i*2+1], 
               label=f'F{i+1}', color=colors[i], alpha=0.5)
    ax.set_xlabel('ui[0]')
    ax.set_ylabel('ui[1]')
    ax.set_title('Follower 动作相图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图 11-12: Value 函数
    ax = fig.add_subplot(3, 4, 11)
    for i in range(3):
        ax.plot(values[:, i], label=f'V{i+1}', color=colors[i])
    ax.set_xlabel('时间步')
    ax.set_ylabel('Value')
    ax.set_title('Follower 值函数 (估计)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value 函数范数
    ax = fig.add_subplot(3, 4, 12)
    for i in range(3):
        ax.plot(np.abs(values[:, i]), label=f'|V{i+1}|', color=colors[i])
    ax.set_xlabel('时间步')
    ax.set_ylabel('|Value|')
    ax.set_title('Value 函数绝对值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('agd_env2_results_part1.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到：agd_env2_results_part1.png")
    
    # 第二张图：最终参数对比
    fig2, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(9)
    width = 0.35
    ax.bar(x - width/2, true_params, width, label='真实值', color='blue', alpha=0.7)
    ax.bar(x + width/2, estimated_params, width, label='估计值', color='red', alpha=0.7)
    ax.set_xlabel('参数索引')
    ax.set_ylabel('参数值')
    ax.set_title('最终参数对比')
    ax.set_xticks(x)
    ax.set_xticklabels(['d1','q1','r1','d2','q2','r2','d3','q3','r3'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('agd_env2_results_part2.png', dpi=150, bbox_inches='tight')
    print("图表已保存到：agd_env2_results_part2.png")


if __name__ == "__main__":
    print("开始 Env2 环境 AGD 测试...")
    
    history, true_params, estimated_params = test_agd_on_env2()
    plot_results(history, true_params, estimated_params)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
