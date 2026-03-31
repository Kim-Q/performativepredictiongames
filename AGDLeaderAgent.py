"""
AGDLeaderAgent - 基于自适应梯度下降的领导者智能体
====================================================

本文档描述如何将 BOLeaderAgent 中的贝叶斯优化方法替换为自适应梯度下降 (AGD) 方法。

核心差异:
----------
BOLeaderAgent:
  - 使用高斯过程 (GP) 建模参数 - 目标函数映射
  - 通过贝叶斯优化 (EI 采集函数) 优化参数 lambda
  - 优点：不需要梯度，自动平衡探索/利用
  - 缺点：计算复杂度 O(n³)，高维空间效率低

AGDLeaderAgent:
  - 使用梯度下降直接优化参数 lambda
  - 通过有限差分或解析梯度更新参数
  - 优点：计算高效 O(n)，易于实现
  - 缺点：需要梯度，需要手动调整学习率

接口说明:
----------
TODO 标记的地方是需要根据具体环境修改的接口。
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from exm2 import BaseAgent, Env2, phi, value, value_gradient, g0_vec, regularized_gram


# ==============================================================================
# AGDLeaderAgent 类
# ==============================================================================

class AGDLeaderAgent(BaseAgent):
    """
    基于自适应梯度下降的领导者智能体。
    
    参数估计目标:
    -------------
    lambda = [
        [d1, q1, r1],  # follower1 的参数
        [d2, q2, r2],  # follower2 的参数
        [d3, q3, r3]   # follower3 的参数
    ]
    
    其中:
    - di: 耦合系数 (follower 如何响应 leader 的动作)
    - qi: 状态代价权重
    - ri: 控制代价权重
    
    估计方法:
    ---------
    1. 预测 follower 动作：u_i_pred = -di * u0 - 0.5 * gi * (gi^T @ ∇Vi)
    2. 计算预测误差：error = ||u_pred - u_actual||^2
    3. 梯度下降更新：lambda += -eta * grad(error)
    
    属性:
    -----
    current_lambda : np.ndarray, shape (3, 3)
        当前估计的 follower 参数 [d, q, r]
    W_hat_followers : List[np.ndarray], shape (3, 5)
        估计的 follower 值函数权重
    """
    
    def __init__(self, env, followers, g_func, theta=0.5, seed=42):
        """
        初始化 AGDLeaderAgent。
        
        参数
        ----
        env : Env2
            环境对象
        followers : List[FollowerAgent]
            跟随者列表（用于获取 gi 函数，但不访问内部参数）
        g_func : Callable
            Leader 的输入向量函数 g0(x)
        theta : float
            折扣因子（默认 0.5）
        seed : int
            随机种子
        """
        super().__init__(env, seed=seed)
        
        self.theta = theta
        self.Q0 = 0.5 * np.eye(2)
        self.C = [0.1, 0.1, 0.1]
        self.R0 = np.eye(2)
        self.followers = followers
        self.g0 = g_func
        
        # ========== 待估计参数 ==========
        # TODO: 根据实际 follower 数量调整形状
        self.current_lambda = np.zeros((3, 3))  # [[d1,q1,r1], [d2,q2,r2], [d3,q3,r3]]
        self.follower_lambda_list = [self.current_lambda.copy()]
        
        # 每个 follower 的值函数权重估计
        n_f = len(followers)
        self.W_hat_followers = [np.zeros(5) for _ in range(n_f)]
        
        # ========== AGD 超参数 ==========
        # TODO: 根据实际问题调整学习率
        self.eta_lambda = 0.01  # lambda 的学习率
        self.eta_W = 0.001      # W_hat 的学习率
        
        # ========== 历史数据 ==========
        self.BO_sample = []  # 兼容 BOLeaderAgent 的命名
        self.state_follower_buffer = []  # [(state, u0, [u1,u2,u3]), ...]
        
        # ========== 梯度估计参数 ==========
        self.eps = 1e-4  # 有限差分扰动
    
    # ==================== 代价函数 ====================
    
    def cost(self, state, *actions):
        """
        领导者代价函数。
        
        J0 = x^T Q0 x + (u0 + Σ ci * ui)^T R0 (u0 + Σ ci * ui)
        
        参数
        ----
        state : np.ndarray, shape (2,)
            当前状态
        *actions : tuple
            (u0, u1, u2, u3) 领导和跟随者的动作
        
        返回
        ----
        cost : float
            代价值
        """
        if len(actions) != 4:
            raise ValueError(f"AGDLeaderAgent.cost 期望 4 个动作，得到 {len(actions)}")
        
        l_action, f1_action, f2_action, f3_action = actions
        
        q1 = state.T @ self.Q0 @ state
        q2 = np.asarray(l_action, dtype=float).copy()
        for ci, fi_action in zip(self.C, [f1_action, f2_action, f3_action]):
            q2 = q2 + float(ci) * np.asarray(fi_action, dtype=float)
        
        r0 = float(q1) + float(q2.T @ self.R0 @ q2)
        return r0
    
    # ==================== 参数估计方法 ====================
    
    def _split_lambda(self, i):
        """提取 follower i 的参数 (di, qi, ri)。"""
        return (
            self.current_lambda[i, 0],  # di
            self.current_lambda[i, 1],  # qi
            self.current_lambda[i, 2]   # ri
        )
    
    def predict_ui_with_lambda(self, state, i):
        """
        使用当前 lambda 估计 follower i 的动作。
        
        u_i = -di * u0 - 0.5 * gi(state) * (gi(state)^T @ ∇Vi)
        
        参数
        ----
        state : np.ndarray
            当前状态
        i : int
            follower 索引
        
        返回
        ----
        u_i_pred : np.ndarray
            预测的 follower 动作
        """
        di, _, _ = self._split_lambda(i)
        
        fi = self.followers[i]
        Wi_hat = self.W_hat_followers[i]
        grad_Vi = value_gradient(state, Wi_hat)
        
        u_i = -di * self.action - 0.5 * fi.gi(state) * (fi.gi(state).T @ grad_Vi)
        return u_i
    
    def compute_gradient(self, state, f_actions):
        """
        计算目标函数关于 lambda 的梯度（有限差分法）。
        
        目标函数：J(λ) = Σ ||u_i_pred(λ) - u_i_actual||^2
        
        参数
        ----
        state : np.ndarray
            当前状态
        f_actions : List[np.ndarray]
            实际观测的 follower 动作
        
        返回
        ----
        grad : np.ndarray, shape (9,)
            梯度向量
        """
        grad = np.zeros(9)
        lambda_flat = self.current_lambda.flatten()
        
        # 当前预测误差
        J_current = self._compute_prediction_error(state, f_actions)
        
        # 有限差分
        for i in range(9):
            lambda_perturbed = lambda_flat.copy()
            lambda_perturbed[i] += self.eps
            
            # 临时更新参数
            self.current_lambda = lambda_perturbed.reshape(3, 3)
            
            # 计算扰动后的误差
            J_perturbed = self._compute_prediction_error(state, f_actions)
            
            # 恢复原参数
            self.current_lambda = lambda_flat.reshape(3, 3)
            
            grad[i] = (J_perturbed - J_current) / self.eps
        
        return grad
    
    def _compute_prediction_error(self, state, f_actions):
        """计算预测误差平方和。"""
        total_mse = 0.0
        for i, u_true in enumerate(f_actions):
            u_pred = self.predict_ui_with_lambda(state, i)
            total_mse += np.linalg.norm(u_pred - np.asarray(u_true)) ** 2
        return total_mse
    
    # ==================== W_hat 估计方法 ====================
    
    def ri_hat(self, state, u0, ui_true, i):
        """
        计算 follower i 的瞬时估计代价。
        
        使用估计的 lambda 参数计算：
        r_i = qi * ||state||^2 + ri * ||ui + di * u0||^2
        
        参数
        ----
        state : np.ndarray
            当前状态
        u0 : np.ndarray
            Leader 动作
        ui_true : np.ndarray
            Follower i 的实际动作
        i : int
            Follower 索引
        
        返回
        ----
        ri : float
            估计的瞬时代价
        """
        di, qi, ri = self._split_lambda(i)
        
        q1 = qi * float(np.asarray(state) @ np.asarray(state))
        coupling_u = np.asarray(ui_true, dtype=float) + di * np.asarray(u0, dtype=float)
        return q1 + ri * float(coupling_u @ coupling_u)
    
    def rhoi_hat(self, state, u0, f_actions, i):
        """
        计算 follower i 的累积代价估计（梯形积分）。
        
        ρ_i = dt * (r_t + r_tT) / 2 + V(state, W_hat_i)
        
        参数
        ----
        state : np.ndarray
            当前状态
        u0 : np.ndarray
            Leader 动作
        f_actions : List[np.ndarray]
            所有 follower 的动作
        i : int
            Follower 索引
        
        返回
        ----
        rho_i : float
            累积代价估计
        """
        ui_true = f_actions[i]
        r_t = self.ri_hat(state, u0, ui_true, i)
        
        u0_arr = np.asarray(u0, dtype=float)
        f_arrs = [np.asarray(u, dtype=float) for u in f_actions]
        next_state = state + self.env.dt * self.env.predict(state, u0_arr, *f_arrs)
        
        r_tT = self.ri_hat(next_state, u0, ui_true, i)
        
        rho_i = 0.5 * self.env.dt * (r_t + r_tT) + value(state, self.W_hat_followers[i])
        return rho_i
    
    def predict_Wi(self, i):
        """
        使用最小二乘法估计 follower i 的值函数权重 W_hat_i。
        
        基于 Bellman 方程：
        ρ_i ≈ φ(state)^T @ W_i
        
        通过历史数据求解：
        min_W Σ ||ρ_i - φ(state)^T W||^2
        
        参数
        ----
        i : int
            Follower 索引
        
        返回
        ----
        W_hat_i : np.ndarray
            估计的权重向量
        """
        if not self.state_follower_buffer:
            return self.W_hat_followers[i]
        
        state_hist = np.array([x[0] for x in self.state_follower_buffer])
        Chi = np.array([phi(s) for s in state_hist])  # (N, 5)
        gram, lambda_ridge = regularized_gram(Chi)
        
        xi_vec = np.array([
            self.rhoi_hat(x[0], x[1], x[2], i)
            for x in self.state_follower_buffer
        ])
        
        rhs = Chi.T @ xi_vec + lambda_ridge * self.W_hat_followers[i]
        self.W_hat_followers[i] = np.linalg.solve(gram, rhs)
        
        return self.W_hat_followers[i]
    
    # ==================== AGD 更新方法 ====================
    
    def update_lambda_agd(self, state, f_actions):
        """
        使用梯度下降更新 lambda 参数。
        
        lambda_new = lambda_old - eta * grad(J)
        
        参数
        ----
        state : np.ndarray
            当前状态
        f_actions : List[np.ndarray]
            实际观测的 follower 动作
        
        返回
        ----
        delta : float
            参数变化量（用于判断收敛）
        """
        # 计算梯度
        grad = self.compute_gradient(state, f_actions)
        
        # 梯度下降更新
        lambda_old = self.current_lambda.copy()
        self.current_lambda = self.current_lambda - self.eta_lambda * grad.reshape(3, 3)
        
        # 参数约束（可选）
        # TODO: 根据实际问题添加参数约束
        self.current_lambda[:, 0] = np.clip(self.current_lambda[:, 0], 0.0, 1.0)  # di in [0, 1]
        self.current_lambda[:, 1:] = np.clip(self.current_lambda[:, 1:], 0.0, 10.0)  # qi, ri > 0
        
        delta = np.linalg.norm(self.current_lambda - lambda_old)
        return delta
    
    def update_W_followers(self):
        """
        更新所有 follower 的 W_hat 估计。
        
        调用 predict_Wi 对每个 follower 进行最小二乘估计。
        """
        for i in range(len(self.followers)):
            self.predict_Wi(i)
    
    # ==================== 动作计算 ====================
    
    def compute_action(self, state: np.ndarray, *f_actions) -> np.ndarray:
        """
        基于估计的参数计算领导者动作。
        
        使用估计的 lambda 和 W_hat 计算最优控制：
        u0* = term1 + term2
        
        其中:
        - term1 与预测误差相关
        - term2 与 follower 的响应相关
        
        参数
        ----
        state : np.ndarray
            当前状态
        *f_actions : tuple
            上一步的 follower 动作 (u1, u2, u3)
        
        返回
        ----
        u0 : np.ndarray
            领导者的最优动作
        """
        if len(f_actions) != 3:
            raise ValueError(f"AGDLeaderAgent.compute_action 需要 3 个 follower 动作")
        
        if len(self.action_hist) == 0:
            # 初始阶段随机探索
            u0 = self.rng.uniform(-1, 1, size=2)
            self.action = u0
            self.action_hist.append(u0.copy())
            self.V.append(value(state, self.W))
            self.compute_rho(state, u0, *f_actions)
            return u0
        
        state = np.asarray(state, dtype=float).reshape(2,)
        f1, f2, f3 = [np.asarray(u, dtype=float).reshape(2,) for u in f_actions]
        
        grad_V0 = value_gradient(state, self.W).reshape(2,)
        
        # 使用估计的 lambda
        d_hat = [float(x) for x in self.current_lambda[:, 0]]
        r_hat = [max(float(x), 1e-6) for x in self.current_lambda[:, 2]]
        
        sum_cd = sum(float(cj) * dj for cj, dj in zip(self.C, d_hat))
        K = (1.0 - sum_cd) * np.eye(2)
        K_reg = K + 1e-6 * np.eye(2)  # 防止奇异
        K_inv = np.linalg.inv(K_reg)
        A0_inv = np.linalg.inv(K_reg.T @ self.R0 @ K_reg)
        
        g0 = self.g0(state)
        sum_gd = np.zeros(2)
        for fj, dj in zip(self.followers, d_hat):
            gj = fj.gi(state)
            sum_gd += gj * dj
        v = sum_gd - g0
        s0 = float(v @ grad_V0)
        term1 = 0.5 * (A0_inv @ (v * s0))
        
        sum2 = np.zeros(2)
        for cj, fj, Wi_hat, rj in zip(self.C, self.followers, self.W_hat_followers, r_hat):
            gj = fj.gi(state)
            grad_Vj = value_gradient(state, Wi_hat)
            sj = float(gj @ grad_Vj)
            sum2 += float(cj) * ((1.0 / max(rj, 1e-6)) * (gj * sj))
        term2 = 0.5 * (K_inv @ sum2)
        
        u0 = term1 + term2
        self.action = u0
        self.action_hist.append(u0.copy())
        self.V.append(value(state, self.W))
        self.compute_rho(state, u0, f1, f2, f3)
        
        return u0
    
    # ==================== 辅助方法 ====================
    
    def ensure_final_weight(self, step_idx: int):
        """确保最后一步的权重被记录（兼容 BaseAgent 接口）。"""
        if self.update_steps[-1] != step_idx:
            self.W_history.append(self.current_lambda.flatten().copy())
            self.update_steps.append(step_idx)
    
    def get_lambda_history(self):
        """获取 lambda 参数历史。"""
        return self.follower_lambda_list


# ==============================================================================
# 使用示例
# ==============================================================================

if __name__ == '__main__':
    """
    使用示例:
    
    from exm2 import Env2, FollowerAgent, g0_vec
    
    # 1. 创建环境
    env = Env2()
    followers = [
        FollowerAgent(env, qi=2.0, ri=1.0, g_func=g1_vec, index=1),
        FollowerAgent(env, qi=1.0, ri=1.0, g_func=g2_vec, index=2),
        FollowerAgent(env, qi=1.5, ri=1.0, g_func=g3_vec, index=3),
    ]
    
    # 2. 创建 AGD Leader
    leader = AGDLeaderAgent(env, followers, g_func=g0_vec)
    
    # 3. 训练循环
    for s in range(max_iters):
        state = env.state.copy()
        
        # Leader 计算动作
        leader_action = leader.compute_action(state, *follower_action)
        
        # Follower 响应
        follower_action = [f.compute_action(state, leader_action) for f in followers]
        
        # 记录历史数据（用于 W_hat 估计）
        leader.state_follower_buffer.append([
            state.copy(), leader_action.copy(),
            [u.copy() for u in follower_action]
        ])
        
        # 更新 W_hat
        leader.update_W_followers()
        
        # 更新 lambda (AGD)
        leader.update_lambda_agd(state, follower_action)
        
        # 环境步进
        env.step(leader_action, *follower_action)
    """
    print(__doc__)
