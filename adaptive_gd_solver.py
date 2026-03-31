"""
自适应梯度下降 (AGD) 求解器 - 用于 Performative Prediction Games
=========================================================================
本模块从 utilsrm.py 中提取 runAGD() 和 update_estimate() 的核心逻辑，
并将其封装为可重用的类，便于迁移到其他博弈环境。

主要特性:
- 通过最小二乘进行在线参数估计
- 注入探索性噪声以实现持续激励
- 自适应学习率调度
- 投影到可行集
- 支持非相关估计 (UNCORR) 和内层迭代 (INNERITER)

迁移指南:
---------
要将此求解器迁移到您的博弈环境，需要实现一个环境类，提供:
1. query_environment(x1, x2) -> (z1, z2): 查询环境获取观测需求
2. sample_base_demand() -> (z1_sample, z2_sample): 从基础需求分布采样

示例 1: 网约车博弈环境 (utilsrm.py)
    class MyGame:
        def query_environment(self, x1, x2):
            # 返回真实观测 (z1, z2)
            return z1_obs, z2_obs
        
        def sample_base_demand(self):
            # 从基础需求分布采样
            return z1_sample, z2_sample
    
    game = MyGame()
    solver = AdaptiveGDSolver(dim=game.dim, ...)
    solver.set_environment_func(game.query_environment, game.sample_base_demand)
    history = solver.run()

示例 2: Env2 环境 (exm2.py) - 领导者 - 跟随者博弈
    from exm2 import Env2, FollowerAgent
    
    env = Env2()
    followers = [FollowerAgent(...), ...]
    
    # 将领导者视为玩家 1，跟随者集体视为玩家 2
    class Env2Wrapper:
        def query_environment(self, leader_action, follower_params):
            # 预测跟随者动作 vs 实际动作
            predicted = predict_follower_actions(leader_action, follower_params)
            actual = [f.compute_action(state, leader_action) for f in followers]
            return predicted, actual
        
        def sample_base_demand(self):
            return env.state.copy(), leader_action
        
    env_wrapper = Env2Wrapper(env, followers)
    solver = AdaptiveGDSolver(dim=3, ...)  # 3 个跟随者参数
    solver.set_environment_func(env_wrapper.query_environment, env_wrapper.sample_base_demand)
    history = solver.run()
"""

import numpy as np
from typing import Tuple, Optional, Callable, List, Dict

class AdaptiveGDSolver:
    """
    通用的自适应梯度下降求解器，用于未知参数的双人博弈。

    博弈结构假设:
    - 玩家 1 的收益取决于 x1, x2 和未知参数 (A1, Ac1)
    - 玩家 2 的收益取决于 x1, x2 和未知参数 (A2, Ac2)
    - 需求/观测模型：z = z_base + A*x_self + Ac*x_other

    参数
    ----------
    dim : int
        决策变量 x1, x2 的维度。
    lam1, lam2 : float
        玩家 1 和 2 的正则化系数。
    nu : float
        参数估计的学习率。
    eta : float
        梯度下降的基础学习率。
    B : float
        探索性噪声的幅度。
    max_iter : int
        最大迭代次数。
    inner_iter : int
        每次梯度步后参数估计的内层迭代次数。
    uncorr : bool
        是否使用非相关估计（仅估计对角线元素）。
    proj_func : Callable, optional
        投影到可行集的函数。默认为恒等映射。
    """

    def __init__(self,
                 dim: int,
                 lam1: float = 0.1,
                 lam2: float = 0.1,
                 nu: float = 0.01,
                 eta: float = 0.001,
                 B: float = 1.0,
                 max_iter: int = 1000,
                 inner_iter: int = 1,
                 uncorr: bool = False,
                 proj_func: Optional[Callable] = None):

        self.dim = dim
        self.lam1 = lam1
        self.lam2 = lam2
        self.nu = nu
        self.eta = eta
        self.B = B
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.uncorr = uncorr

        # 默认投影：恒等映射（无约束）
        self.proj = proj_func if proj_func else lambda x: x

        # 单位矩阵（用于梯度计算）
        self.I = np.eye(dim)

        # 初始化决策变量
        self.x1 = np.zeros(dim)
        self.x2 = np.zeros(dim)

        # 初始化参数估计（零初始化）
        self.A1_hat = np.zeros((dim, dim))
        self.Ac1_hat = np.zeros((dim, dim))
        self.A2_hat = np.zeros((dim, dim))
        self.Ac2_hat = np.zeros((dim, dim))

        # 环境接口（需要用户设置）
        self._env_query_func: Optional[Callable] = None  # (x1, x2) -> (z1, z2)
        self._sample_base_func: Optional[Callable] = None  # () -> (z1_sample, z2_sample)

        # 轨迹存储
        self.history = {
            'x1': [], 'x2': [],
            'A1_hat': [], 'Ac1_hat': [],
            'A2_hat': [], 'Ac2_hat': [],
            'grad_norm': [],
            'eta': []
        }

    def set_environment_func(self, 
                              env_query_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                              sample_base_func: Callable[[], Tuple[np.ndarray, np.ndarray]]):
        """
        设置环境查询接口。

        这是迁移到真实博弈环境的关键方法。

        参数
        ----------
        env_query_func : Callable
            环境查询函数，签名：func(x1, x2) -> (z1, z2)
            返回带噪声的真实观测需求。
        sample_base_func : Callable
            基础需求采样函数，签名：func() -> (z1_sample, z2_sample)
            从基础需求分布采样（用于梯度计算）。

        示例
        -------
        >>> def my_env_query(x1, x2):
        ...     z1 = self.z1_base + self.A1_true @ x1 + self.Ac1_true @ x2 + noise
        ...     z2 = self.z2_base + self.A2_true @ x2 + self.Ac2_true @ x1 + noise
        ...     return z1, z2
        >>> 
        >>> def my_sample_base():
        ...     return self.z1_base + small_noise, self.z2_base + small_noise
        >>> 
        >>> solver.set_environment_func(my_env_query, my_sample_base)
        """
        self._env_query_func = env_query_func
        self._sample_base_func = sample_base_func

    def reset(self):
        """重置求解器状态。"""
        self.x1 = np.zeros(self.dim)
        self.x2 = np.zeros(self.dim)
        self.A1_hat = np.zeros((self.dim, self.dim))
        self.Ac1_hat = np.zeros((self.dim, self.dim))
        self.A2_hat = np.zeros((self.dim, self.dim))
        self.Ac2_hat = np.zeros((self.dim, self.dim))
        self.history = {
            'x1': [], 'x2': [],
            'A1_hat': [], 'Ac1_hat': [],
            'A2_hat': [], 'Ac2_hat': [],
            'grad_norm': [],
            'eta': []
        }

    def get_gradient(self, x1: np.ndarray, x2: np.ndarray,
                     z1_sample: np.ndarray, z2_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用估计的参数计算纳什均衡梯度。

        梯度公式（与原始 utilsrm.py 中的 getgrad_adaptive 一致）:
        p1 = -(A1_hat - lam1*I).T @ x1 - 0.5 * (z1_sample + Ac1_hat @ x2)
        p2 = -(A2_hat - lam2*I).T @ x2 - 0.5 * (z2_sample + Ac2_hat @ x1)

        参数
        -------
        x1, x2 : np.ndarray
            玩家 1 和 2 的当前决策。
        z1_sample, z2_sample : np.ndarray
            从基础需求分布采样的需求（用于梯度计算）。

        返回
        -------
        grad1, grad2 : np.ndarray
            玩家 1 和 2 的梯度。
        """
        # 玩家 1 梯度
        grad1 = -(self.A1_hat - self.lam1 * self.I).T @ x1 - 0.5 * (z1_sample + self.Ac1_hat @ x2)

        # 玩家 2 梯度
        grad2 = -(self.A2_hat - self.lam2 * self.I).T @ x2 - 0.5 * (z2_sample + self.Ac2_hat @ x1)

        return grad1, grad2

    def update_parameters(self, x1: np.ndarray, x2: np.ndarray,
                          z1_sample: np.ndarray, z2_sample: np.ndarray):
        """
        使用最小二乘更新参数估计。

        这是对原始 utilsrm.py 中 update_estimate 方法的改进版本，
        修正了原始代码中使用真实参数而非估计参数的问题。

        更新规则:
        barA_hat_new = barA_hat + nu * (q - z_pred - barA_hat @ v) @ v.T

        其中:
        - v = [u1; u2] 是探索噪声向量
        - q 是从真实环境获取的观测
        - z_pred = z_sample + A_hat@x1 + Ac_hat@x2 是使用估计参数的预测

        关键洞察:
        残差 = q - z_pred - barA_hat @ v
             = (z_sample + A_true@(x1+u1) + Ac_true@(x2+u2) + noise) 
               - (z_sample + A_hat@x1 + Ac_hat@x2) 
               - (A_hat@u1 + Ac_hat@u2)
             = (A_true - A_hat)@x1 + (Ac_true - Ac_hat)@x2 + noise
             ≈ 参数误差的线性组合

        参数
        ----------
        x1, x2 : np.ndarray
            玩家 1 和 2 的当前决策。
        z1_sample, z2_sample : np.ndarray
            从基础需求分布采样的需求。
        """
        # --- 生成探索噪声 ---
        u1 = self.B * np.random.randn(self.dim)
        u2 = self.B * np.random.randn(self.dim)

        # 拼接噪声向量 v = [u1; u2]
        v = np.concatenate([u1, u2])  # 形状 (2*dim,)
        v_temp = v.reshape(-1, 1)     # 形状 (2*dim, 1)

        # --- 获取环境观测 ---
        if self._env_query_func is not None:
            # 使用设置的环境接口获取真实观测
            q1, q2 = self._env_query_func(x1 + u1, x2 + u2)
        else:
            raise RuntimeError("必须设置环境接口才能更新参数")

        # --- 玩家 1 参数估计 ---
        # 使用估计参数的预测：z1_pred = z1_sample + A1_hat@x1 + Ac1_hat@x2
        z1_pred = z1_sample + self.A1_hat @ x1 + self.Ac1_hat @ x2

        # 拼接参数矩阵 barA1 = [A1, Ac1]
        if self.uncorr:
            barA1_hat = np.hstack([np.diag(np.diag(self.A1_hat)),
                                   np.diag(np.diag(self.Ac1_hat))])
        else:
            barA1_hat = np.hstack([self.A1_hat, self.Ac1_hat])

        # SGD 更新：barA1_hat_new = barA1_hat + nu * (q1 - z1_pred - barA1_hat @ v) @ v.T
        residual1 = q1.reshape(-1, 1) - z1_pred.reshape(-1, 1) - barA1_hat @ v_temp
        barA1_hat_new = barA1_hat + self.nu * (residual1 @ v_temp.T)

        # 拆分回 A1 和 Ac1
        if self.uncorr:
            self.A1_hat = np.diag(np.diag(barA1_hat_new[:, :self.dim]))
            self.Ac1_hat = np.diag(np.diag(barA1_hat_new[:, self.dim:]))
        else:
            self.A1_hat = barA1_hat_new[:, :self.dim]
            self.Ac1_hat = barA1_hat_new[:, self.dim:]

        # --- 玩家 2 参数估计 ---
        # 注意：玩家 2 的噪声拼接顺序是 [u2; u1]
        v_ = np.concatenate([u2, u1])
        v_temp_ = v_.reshape(-1, 1)

        z2_pred = z2_sample + self.A2_hat @ x2 + self.Ac2_hat @ x1

        if self.uncorr:
            barA2_hat = np.hstack([np.diag(np.diag(self.A2_hat)),
                                   np.diag(np.diag(self.Ac2_hat))])
        else:
            barA2_hat = np.hstack([self.A2_hat, self.Ac2_hat])

        residual2 = q2.reshape(-1, 1) - z2_pred.reshape(-1, 1) - barA2_hat @ v_temp_
        barA2_hat_new = barA2_hat + self.nu * (residual2 @ v_temp_.T)

        if self.uncorr:
            self.A2_hat = np.diag(np.diag(barA2_hat_new[:, :self.dim]))
            self.Ac2_hat = np.diag(np.diag(barA2_hat_new[:, self.dim:]))
        else:
            self.A2_hat = barA2_hat_new[:, :self.dim]
            self.Ac2_hat = barA2_hat_new[:, self.dim:]

        # 稳定性：裁剪参数矩阵到合理范围
        self.A1_hat = np.clip(self.A1_hat, -50, 50)
        self.Ac1_hat = np.clip(self.Ac1_hat, -50, 50)
        self.A2_hat = np.clip(self.A2_hat, -50, 50)
        self.Ac2_hat = np.clip(self.Ac2_hat, -50, 50)

    def step(self, t: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        执行一次自适应梯度下降迭代（需要已设置环境接口）。

        参数
        ----------
        t : int
            当前迭代索引。

        返回
        -------
        x1, x2 : np.ndarray
            更新后的决策。
        info : dict
            包含当前估计和梯度的字典。
        """
        if self._sample_base_func is None:
            raise RuntimeError("请先调用 set_environment_func() 设置环境接口")

        # 1. 从基础需求分布采样
        z1_sample, z2_sample = self._sample_base_func()

        # 2. 更新参数估计（内层迭代）
        for _ in range(self.inner_iter):
            self.update_parameters(self.x1, self.x2, z1_sample, z2_sample)

        # 3. 计算梯度
        grad1, grad2 = self.get_gradient(self.x1, self.x2, z1_sample, z2_sample)

        # 4. 自适应学习率：eta / log(t + 2)
        eta_t = self.eta / np.log(t + 2)

        # 5. 梯度下降步 + 投影
        self.x1 = self.proj(self.x1 - eta_t * grad1)
        self.x2 = self.proj(self.x2 - eta_t * grad2)

        # 记录历史
        self.history['x1'].append(self.x1.copy())
        self.history['x2'].append(self.x2.copy())
        self.history['A1_hat'].append(self.A1_hat.copy())
        self.history['Ac1_hat'].append(self.Ac1_hat.copy())
        self.history['A2_hat'].append(self.A2_hat.copy())
        self.history['Ac2_hat'].append(self.Ac2_hat.copy())
        self.history['grad_norm'].append(np.linalg.norm(np.concatenate([grad1, grad2])))
        self.history['eta'].append(eta_t)

        info = {
            'grad_norm': np.linalg.norm(np.concatenate([grad1, grad2])),
            'eta': eta_t
        }

        return self.x1, self.x2, info

    def step_with_samples(self, t: int, z1_sample: np.ndarray, z2_sample: np.ndarray,
                          env_query_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        使用提供的样本执行一次迭代（手动模式）。

        此方法允许您在没有设置环境接口的情况下手动控制每一步。

        参数
        ----------
        t : int
            当前迭代索引。
        z1_sample, z2_sample : np.ndarray
            从基础需求分布采样的需求。
        env_query_func : Callable, optional
            临时的环境查询函数。如果提供，会覆盖已设置的接口。

        返回
        -------
        x1, x2 : np.ndarray
            更新后的决策。
        info : dict
            包含当前估计和梯度的字典。
        """
        # 临时设置环境查询函数（如果提供）
        old_env_func = self._env_query_func
        if env_query_func is not None:
            self._env_query_func = env_query_func

        # 2. 更新参数估计（内层迭代）
        for _ in range(self.inner_iter):
            self.update_parameters(self.x1, self.x2, z1_sample, z2_sample)

        # 3. 计算梯度
        grad1, grad2 = self.get_gradient(self.x1, self.x2, z1_sample, z2_sample)

        # 4. 自适应学习率
        eta_t = self.eta / np.log(t + 2)

        # 5. 梯度下降步 + 投影
        self.x1 = self.proj(self.x1 - eta_t * grad1)
        self.x2 = self.proj(self.x2 - eta_t * grad2)

        # 恢复原来的环境函数
        self._env_query_func = old_env_func

        # 记录历史
        self.history['x1'].append(self.x1.copy())
        self.history['x2'].append(self.x2.copy())
        self.history['A1_hat'].append(self.A1_hat.copy())
        self.history['Ac1_hat'].append(self.Ac1_hat.copy())
        self.history['A2_hat'].append(self.A2_hat.copy())
        self.history['Ac2_hat'].append(self.Ac2_hat.copy())
        self.history['grad_norm'].append(np.linalg.norm(np.concatenate([grad1, grad2])))
        self.history['eta'].append(eta_t)

        info = {
            'grad_norm': np.linalg.norm(np.concatenate([grad1, grad2])),
            'eta': eta_t
        }

        return self.x1, self.x2, info

    def run(self, eta_base: Optional[float] = None,
            nu: Optional[float] = None,
            verbose: bool = True) -> dict:
        """
        运行完整的 AGD 优化循环（需要已设置环境接口）。

        参数
        ----------
        eta_base : float, optional
            基础学习率（覆盖初始化时的值）。
        nu : float, optional
            参数估计学习率（覆盖初始化时的值）。
        verbose : bool
            是否打印进度信息。

        返回
        -------
        history : dict
            包含完整轨迹历史的字典。
        """
        if self._sample_base_func is None:
            raise RuntimeError("请先调用 set_environment_func() 设置环境接口")

        self.reset()

        # 可选：覆盖学习率参数
        if eta_base is not None:
            self.eta = eta_base
        if nu is not None:
            self.nu = nu

        if verbose:
            print(f"开始 AGD，共 {self.max_iter} 次迭代...")
            print(f"  基础学习率 eta = {self.eta}, 参数学习率 nu = {self.nu}")
            print(f"  探索噪声 B = {self.B}, 内层迭代 = {self.inner_iter}")

        for t in range(self.max_iter):
            x1, x2, info = self.step(t)

            if verbose and (t % max(1, self.max_iter // 10) == 0 or t == self.max_iter - 1):
                print(f"迭代 {t}: 梯度范数={info['grad_norm']:.4e}, 学习率={info['eta']:.4e}")

        if verbose:
            print("AGD 完成。")

        return self.history

    def run_with_manual_samples(self, z1_base: np.ndarray, z2_base: np.ndarray,
                                 env_query_func: Callable,
                                 eta_base: Optional[float] = None,
                                 nu: Optional[float] = None,
                                 verbose: bool = True) -> dict:
        """
        使用手动提供的 z_base 运行 AGD（简化接口）。

        此方法适用于测试场景，其中 z_base 是固定的或简单采样的。

        参数
        ----------
        z1_base, z2_base : np.ndarray
            基础需求向量。
        env_query_func : Callable
            环境查询函数，签名：func(x1, x2) -> (z1, z2)。
        eta_base : float, optional
            基础学习率。
        nu : float, optional
            参数估计学习率。
        verbose : bool
            是否打印进度信息。

        返回
        -------
        history : dict
            包含完整轨迹历史的字典。
        """
        self.reset()

        if eta_base is not None:
            self.eta = eta_base
        if nu is not None:
            self.nu = nu

        if verbose:
            print(f"开始 AGD，共 {self.max_iter} 次迭代...")
            print(f"  基础学习率 eta = {self.eta}, 参数学习率 nu = {self.nu}")
            print(f"  探索噪声 B = {self.B}, 内层迭代 = {self.inner_iter}")

        for t in range(self.max_iter):
            # 从 z_base 采样（添加小型噪声）
            z1_sample = z1_base + np.random.randn(self.dim) * 0.01
            z2_sample = z2_base + np.random.randn(self.dim) * 0.01

            x1, x2, info = self.step_with_samples(t, z1_sample, z2_sample, env_query_func)

            if verbose and (t % max(1, self.max_iter // 10) == 0 or t == self.max_iter - 1):
                print(f"迭代 {t}: 梯度范数={info['grad_norm']:.4e}, 学习率={info['eta']:.4e}")

        if verbose:
            print("AGD 完成。")

        return self.history

    def set_initial_params(self, A1_hat: np.ndarray, Ac1_hat: np.ndarray,
                           A2_hat: np.ndarray, Ac2_hat: np.ndarray):
        """
        设置初始参数估计。

        参数
        ----------
        A1_hat, Ac1_hat : np.ndarray
            玩家 1 的参数估计。
        A2_hat, Ac2_hat : np.ndarray
            玩家 2 的参数估计。
        """
        self.A1_hat = A1_hat.copy()
        self.Ac1_hat = Ac1_hat.copy()
        self.A2_hat = A2_hat.copy()
        self.Ac2_hat = Ac2_hat.copy()

    def set_initial_position(self, x1: np.ndarray, x2: np.ndarray):
        """
        设置初始决策位置。

        参数
        ----------
        x1, x2 : np.ndarray
            初始决策。
        """
        self.x1 = x1.copy()
        self.x2 = x2.copy()
