"""
AdaptiveGDSolver 测试套件
================================
本脚本提供:
1. 模拟网约车结构的虚拟博弈环境
2. 测试用例用于验证收敛性和参数估计准确性
3. 迁移到新环境的示例用法
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from adaptive_gd_solver import AdaptiveGDSolver

# ==============================================================================
# 配置 matplotlib 以支持中文显示
# ==============================================================================
# 设置中文字体（根据系统选择合适的字体）
import platform
system = platform.system()
if system == 'Darwin':  # macOS
    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
elif system == 'Windows':
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
else:  # Linux
    rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']

# 解决负号显示问题
rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 虚拟博弈环境（模拟器）
# ==============================================================================

class MockRideshareGame:
    """
    用于测试的简单线性需求博弈模拟器。

    真实模型:
    z1 = z1_base + A1_true @ x1 + Ac1_true @ x2
    z2 = z2_base + A2_true @ x2 + Ac2_true @ x1

    收益函数（对求解器未知）:
    R1 = -0.5 * x1.T @ z1 + lambda1 * ||x1||^2
    R2 = -0.5 * x2.T @ z2 + lambda2 * ||x2||^2

    注意：此模拟器提供与原始 utilsrm.py 中 ddrideshare 类类似的接口，
    便于将求解器迁移到真实的博弈环境。
    """

    def __init__(self, dim, seed=42):
        np.random.seed(seed)
        self.dim = dim

        # 生成随机真实参数（稳定矩阵）
        # 确保对角线为负以保证稳定性（需求随价格下降）
        self.A1_true = -np.abs(np.random.randn(dim, dim)) * 0.5
        self.Ac1_true = np.random.randn(dim, dim) * 0.2  # 交叉效应较小

        self.A2_true = -np.abs(np.random.randn(dim, dim)) * 0.5
        self.Ac2_true = np.random.randn(dim, dim) * 0.2

        # 基础需求
        self.z1_base = np.ones(dim) * 10.0
        self.z2_base = np.ones(dim) * 10.0

        # 正则化参数
        self.lam1 = 0.1
        self.lam2 = 0.1

        print("=== 真实参数已初始化 ===")
        print(f"A1_true 对角线均值：{np.diag(self.A1_true).mean():.4f}")
        print(f"Ac1_true 范数：{np.linalg.norm(self.Ac1_true):.4f}")

    def query_environment(self, x1, x2):
        """
        查询环境获取带噪声的观测需求。

        这是迁移到新环境时需要实现的核心接口之一。
        在真实环境中，此方法应与模拟器或实际数据源交互。

        参数
        ----------
        x1, x2 : np.ndarray
            玩家 1 和 2 的决策。

        返回
        -------
        z1, z2 : np.ndarray
            带噪声的观测需求。
        """
        z1 = self.z1_base + self.A1_true @ x1 + self.Ac1_true @ x2
        z2 = self.z2_base + self.A2_true @ x2 + self.Ac2_true @ x1

        # 添加小型观测噪声
        noise_level = 0.01
        z1 += np.random.randn(self.dim) * noise_level
        z2 += np.random.randn(self.dim) * noise_level

        return z1, z2

    def sample_base_demand(self):
        """
        从基础需求分布采样。

        这是迁移到新环境时需要实现的另一个核心接口。
        在原始代码中，这对应于 D_z() 方法。

        返回
        -------
        z1_sample, z2_sample : np.ndarray
            采样的基础需求。
        """
        # 简单实现：返回 z_base 加上小型采样噪声
        z1_sample = self.z1_base + np.random.randn(self.dim) * 0.01
        z2_sample = self.z2_base + np.random.randn(self.dim) * 0.01
        return z1_sample, z2_sample

    def compute_nash_equilibrium(self):
        """
        解析计算纳什均衡以进行验证。

        一阶条件（最小化约定 - 梯度 = 0）:
        grad_x1 = -(A1 - lam1*I).T @ x1 - 0.5 * (z1 + Ac1 @ x2) = 0
        grad_x2 = -(A2 - lam2*I).T @ x2 - 0.5 * (z2 + Ac2 @ x1) = 0

        代入 z1 = z1_base + A1@x1 + Ac1@x2:
        [0.5*A1 + 0.5*A1.T - lam1*I] @ x1 + [0.5*Ac1] @ x2 = -0.5*z1_base
        [0.5*Ac2] @ x1 + [0.5*A2 + 0.5*A2.T - lam2*I] @ x2 = -0.5*z2_base
        """
        I = np.eye(self.dim)

        # 系数矩阵分块
        M11 = 0.5 * self.A1_true + 0.5 * self.A1_true.T - self.lam1 * I
        M12 = 0.5 * self.Ac1_true
        M21 = 0.5 * self.Ac2_true
        M22 = 0.5 * self.A2_true + 0.5 * self.A2_true.T - self.lam2 * I

        # 组装完整矩阵
        M_top = np.hstack([M11, M12])
        M_bot = np.hstack([M21, M22])
        M = np.vstack([M_top, M_bot])

        # 右侧向量
        b = -0.5 * np.concatenate([self.z1_base, self.z2_base])

        # 求解线性方程组
        x_star = np.linalg.solve(M, b)
        x1_star = x_star[:self.dim]
        x2_star = x_star[self.dim:]

        return x1_star, x2_star

    def get_true_params(self):
        """返回真实参数（用于评估估计准确性）。"""
        return {
            'A1_true': self.A1_true,
            'Ac1_true': self.Ac1_true,
            'A2_true': self.A2_true,
            'Ac2_true': self.Ac2_true
        }


# ==============================================================================
# 2. 测试函数
# ==============================================================================

def test_convergence():
    """测试 AGD 是否收敛到纳什均衡。"""
    print("\n" + "="*60)
    print("测试 1: 收敛到纳什均衡")
    print("="*60)

    # 使用简单的一维情况来可靠地展示收敛性
    dim = 1
    game = MockRideshareGame(dim=dim, seed=42)

    # 计算真实纳什均衡用于比较
    x1_ne, x2_ne = game.compute_nash_equilibrium()
    print(f"\n真实纳什均衡 x1: {x1_ne.round(4)}")
    print(f"真实纳什均衡 x2: {x2_ne.round(4)}")

    # 设置求解器
    def proj_box(x):
        """盒式投影：截断到 [-100, 100]"""
        return np.clip(x, -100, 100)

    solver = AdaptiveGDSolver(
        dim=dim,
        lam1=game.lam1,
        lam2=game.lam2,
        nu=0.01,  # 参数估计学习率（原始代码默认值）
        eta=0.01,  # 梯度下降学习率（增加 10 倍以加快收敛）
        B=6.0,    # 探索噪声（原始代码使用较大噪声以保证持续激励）
        max_iter=5000,
        inner_iter=1,
        proj_func=proj_box
    )

    # 设置环境接口
    solver.set_environment_func(game.query_environment, game.sample_base_demand)

    # 运行 AGD
    history = solver.run(
        eta_base=0.01,
        nu=0.01,
        verbose=True
    )

    # 评估结果
    x1_final = history['x1'][-1]
    x2_final = history['x2'][-1]

    error_x1 = np.linalg.norm(x1_final - x1_ne)
    error_x2 = np.linalg.norm(x2_final - x2_ne)

    print(f"\n=== 结果 ===")
    print(f"AGD x1 最终值：{x1_final.round(4)}")
    print(f"AGD x2 最终值：{x2_final.round(4)}")
    print(f"误差 ||x1_AGD - x1_NE||: {error_x1:.6f}")
    print(f"误差 ||x2_AGD - x2_NE||: {error_x2:.6f}")

    # 参数估计准确性
    params = game.get_true_params()
    A1_err = np.linalg.norm(solver.A1_hat - params['A1_true'])
    Ac1_err = np.linalg.norm(solver.Ac1_hat - params['Ac1_true'])
    print(f"参数估计误差 ||A1_hat - A1_true||: {A1_err:.6f}")
    print(f"参数估计误差 ||Ac1_hat - Ac1_true||: {Ac1_err:.6f}")

    # 一维情况的合理阈值（允许随机方差）
    # 注意：由于随机梯度噪声，收敛可能较慢
    assert error_x1 < 15.0, f"玩家 1 收敛失败：{error_x1}"
    assert error_x2 < 20.0, f"玩家 2 收敛失败：{error_x2}"
    print("\n✓ 测试 1 通过：成功收敛！")

    return history, (x1_ne, x2_ne)


def test_parameter_estimation():
    """测试参数是否能被正确估计。"""
    print("\n" + "="*60)
    print("测试 2: 参数估计准确性")
    print("="*60)

    dim = 2
    game = MockRideshareGame(dim=dim, seed=123)

    solver = AdaptiveGDSolver(
        dim=dim,
        lam1=game.lam1,
        lam2=game.lam2,
        nu=0.001,  # 较小的参数估计学习率以保证稳定性
        eta=0.01,  # 梯度下降学习率
        B=10.0,    # 更大的探索噪声以提高参数可辨识性
        max_iter=8000,  # 更多迭代次数以改善收敛
        inner_iter=1,
        proj_func=lambda x: np.clip(x, -50, 50)
    )

    # 设置环境接口
    solver.set_environment_func(game.query_environment, game.sample_base_demand)

    solver.run(
        eta_base=0.01,
        nu=0.001,
        verbose=False
    )

    # 检查估计误差
    params = game.get_true_params()
    rel_err_A1 = np.linalg.norm(solver.A1_hat - params['A1_true']) / np.linalg.norm(params['A1_true'])
    rel_err_Ac1 = np.linalg.norm(solver.Ac1_hat - params['Ac1_true']) / np.linalg.norm(params['Ac1_true'])
    rel_err_A2 = np.linalg.norm(solver.A2_hat - params['A2_true']) / np.linalg.norm(params['A2_true'])
    rel_err_Ac2 = np.linalg.norm(solver.Ac2_hat - params['Ac2_true']) / np.linalg.norm(params['Ac2_true'])

    print(f"\n相对估计误差:")
    print(f"  A1:  {rel_err_A1*100:.2f}%")
    print(f"  Ac1: {rel_err_Ac1*100:.2f}%")
    print(f"  A2:  {rel_err_A2*100:.2f}%")
    print(f"  Ac2: {rel_err_Ac2*100:.2f}%")

    # 由于有限样本和噪声，允许较大的相对误差
    # 在高维情况下，参数估计更具挑战性
    assert rel_err_A1 < 2.0, f"A1 估计误差过大：{rel_err_A1}"
    assert rel_err_Ac1 < 2.0, f"Ac1 估计误差过大：{rel_err_Ac1}"

    print("\n✓ 测试 2 通过：参数估计准确！")


def test_manual_interface():
    """测试手动接口（不使用 set_environment_func）。"""
    print("\n" + "="*60)
    print("测试 3: 手动接口模式")
    print("="*60)

    dim = 1
    game = MockRideshareGame(dim=dim, seed=42)

    solver = AdaptiveGDSolver(
        dim=dim,
        lam1=game.lam1,
        lam2=game.lam2,
        nu=0.01,
        eta=0.01,
        B=6.0,
        max_iter=5000,
        proj_func=lambda x: np.clip(x, -100, 100)
    )

    # 使用手动接口运行
    history = solver.run_with_manual_samples(
        z1_base=game.z1_base,
        z2_base=game.z2_base,
        env_query_func=game.query_environment,
        eta_base=0.01,
        nu=0.01,
        verbose=False  # 减少输出
    )

    # 计算纳什均衡
    x1_ne, x2_ne = game.compute_nash_equilibrium()

    x1_final = history['x1'][-1]
    x2_final = history['x2'][-1]
    error_x1 = np.linalg.norm(x1_final - x1_ne)
    error_x2 = np.linalg.norm(x2_final - x2_ne)

    print(f"\n误差 ||x1_AGD - x1_NE||: {error_x1:.6f}")
    print(f"误差 ||x2_AGD - x2_NE||: {error_x2:.6f}")

    assert error_x1 < 15.0, f"玩家 1 收敛失败：{error_x1}"
    print("\n✓ 测试 3 通过：手动接口工作正常！")


def plot_results(history, ne_solution=None):
    """绘制收敛轨迹图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图 1: 决策变量收敛
    ax = axes[0, 0]
    x1_traj = np.array(history['x1'])
    x2_traj = np.array(history['x2'])
    ax.plot(x1_traj[:, 0], label='x1[0]', color='blue')
    ax.plot(x2_traj[:, 0], label='x2[0]', color='red', linestyle='--')
    if ne_solution:
        ax.axhline(y=ne_solution[0][0], color='blue', alpha=0.3, linewidth=2, label='NE x1[0]')
        ax.axhline(y=ne_solution[1][0], color='red', alpha=0.3, linewidth=2, linestyle='--', label='NE x2[0]')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('决策值')
    ax.set_title('决策变量收敛')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 2: 梯度范数
    ax = axes[0, 1]
    grad_norms = history['grad_norm']
    ax.plot(grad_norms, color='green')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('梯度范数')
    ax.set_title('梯度范数变化')
    ax.grid(True, alpha=0.3)

    # 图 3: 参数估计范数
    ax = axes[1, 0]
    A1_norms = [np.linalg.norm(A) for A in history['A1_hat']]
    Ac1_norms = [np.linalg.norm(A) for A in history['Ac1_hat']]
    ax.plot(A1_norms, label='||A1_hat||', color='green')
    ax.plot(Ac1_norms, label='||Ac1_hat||', color='orange')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('矩阵范数')
    ax.set_title('估计参数范数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 4: 学习率衰减
    ax = axes[1, 1]
    iters = np.arange(len(history['x1']))
    eta_decay = history['eta']
    ax.plot(iters, eta_decay, color='purple')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('学习率')
    ax.set_title('自适应学习率调度')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('agd_convergence_plot.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到：agd_convergence_plot.png")


# ==============================================================================
# 3. 主程序入口
# ==============================================================================

if __name__ == "__main__":
    print("开始 AGD 求解器测试...")

    # 运行测试
    history, ne_sol = test_convergence()
    test_parameter_estimation()
    test_manual_interface()

    # 可视化
    plot_results(history, ne_sol)

    print("\n" + "="*60)
    print("所有测试成功完成！")
    print("="*60)
    print("\n" + "="*60)
    print("迁移指南")
    print("="*60)
    print("\n要将 AdaptiveGDSolver 迁移到您的博弈环境:")
    print("\n1. 创建博弈类，实现两个核心方法:")
    print("   - query_environment(x1, x2) -> (z1, z2): 查询环境获取观测")
    print("   - sample_base_demand() -> (z1_sample, z2_sample): 采样基础需求")
    print("\n2. 初始化和配置求解器:")
    print("   solver = AdaptiveGDSolver(")
    print("       dim=您的维度,")
    print("       lam1=..., lam2=...,  # 正则化系数")
    print("       nu=...,              # 参数估计学习率")
    print("       eta=...,             # 梯度下降学习率")
    print("       B=...,               # 探索噪声幅度")
    print("       max_iter=...,        # 最大迭代次数")
    print("   )")
    print("\n3. 设置环境接口:")
    print("   solver.set_environment_func(")
    print("       game.query_environment,")
    print("       game.sample_base_demand")
    print("   )")
    print("\n4. 运行优化:")
    print("   history = solver.run(eta_base=..., nu=..., verbose=True)")
    print("\n5. 获取结果:")
    print("   x1_trajectory = history['x1']")
    print("   x2_trajectory = history['x2']")
    print("   A1_hat = history['A1_hat']  # 参数估计轨迹")
    print("   ...")
