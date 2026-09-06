import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix


physics_params = {
    # ---------- 几何与流动（不可修改） ----------
    'L': 2.0,               # 通道长度 [m]
    'H': 1.0,               # 通道高度 [m]
    'U_max': 0.2,           # 最大速度 [m/s]

    # ---------- 中子物理 ----------
    'D': 0.1,              # 中子扩散系数 [m]，强扩散抹平局部 S_ext 峰值
    'Sigma_a0': 1.6,        # 参考宏观吸收截面 [m^-1]
    'beta': 0.0065,           # 缓发中子份额 [-]，提高 C 产生
    'nu': 2.5,              # 每次裂变释放中子数 [-]
    'Sigma_f': 0.6,         # 宏观裂变截面 [m^-1]
    'nu_Sigma_f': 1.5,      # nu * Sigma_f [m^-1]
    'lambda': 0.1,         # 先驱核衰变常数 [s^-1]

    # ---------- 热工水力 ----------
    'T0': 600.0,            # 参考温度 [K]
    'rho_Cp': 1000.0,       # 体积比热容 [J/(m^3 K)]
    'k_th': 5.0,            # 热导率 [W/(m K)]
    'alpha_T': 1.0e-3,      # 温度反馈系数 [K^-1]，温和反馈
    'epsilon_f': 1.0e-2,    # 每次裂变释放能量 [J]

    # ---------- 先驱核 ----------
    'D_c': 0.01,           # 先驱核扩散系数 [m^2/s]

    # ---------- 外中子源（固定参数，宽平台分布） ----------
    'S0': 1.0e8,            # 外源强度 [s^-1 m^-3]
    'x0': 1.1,              # 外源中心 x [m]
    'y0': 0.4,              # 外源中心 y [m]
    'sigma_x': 0.2,         # 外源 x 方向标准差 [m]，宽平台
    'sigma_y': 0.1,         # 外源 y 方向标准差 [m]，宽平台
}


def solve_2d_phi_2_T(phi, nx=201, ny=101, T_inlet=600.0):
    """
    求解二维稳态热传导-平流方程:
        rho*Cp * u(y) * dT/dx = k_th * (d2T/dx2 + d2T/dy2) + q'''
    其中 q''' = epsilon_f * Sigma_f * phi(x, y)

    默认边界条件：
        - 左边界 (x=0): Dirichlet，T = T_inlet
        - 右边界 (x=L): Neumann，dT/dx = 0
        - 下边界 (y=0): Neumann，dT/dy = 0
        - 上边界 (y=H): Neumann，dT/dy = 0

    phi 可以是 2D numpy 数组 (ny, nx) 或 callable(x_vec, y_vec)。
    返回 X, Y, T 三个 2D 数组，以及 u 速度剖面 1D 数组。
    """
    L = physics_params['L']
    H = physics_params['H']
    U_max = physics_params['U_max']
    rho_Cp = physics_params['rho_Cp']
    k_th = physics_params['k_th']
    epsilon_f = physics_params['epsilon_f']
    Sigma_f = physics_params['Sigma_f']

    dx = L / (nx - 1)
    dy = H / (ny - 1)

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    # Poiseuille 速度剖面，u >= 0
    u = U_max * 4.0 * (y / H) * (1.0 - y / H)

    # 构造热源
    if callable(phi):
        phi_vals = phi(X, Y)
    else:
        if phi.shape != (ny, nx):
            raise ValueError(f"phi shape {phi.shape} 与网格 (ny={ny}, nx={nx}) 不一致")
        phi_vals = phi
    q = epsilon_f * Sigma_f * phi_vals

    N = nx * ny

    def idx(i, j):
        return i * nx + j

    cx = k_th / (dx ** 2)
    cy = k_th / (dy ** 2)

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)

            # 左边界 Dirichlet
            if j == 0:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = T_inlet
                continue

            # 右端热源项：A T = q
            b[k] = q[i, j]

            # 扩散 Laplacian 对角项（移到左边后为正）
            rows.append(k); cols.append(k); vals.append(2.0 * cx + 2.0 * cy)

            # --- X 方向扩散 ---
            if j == nx - 1:
                # 右边界 Neumann：T_{j+1} = T_{j-1}
                if j - 1 == 0:
                    b[k] += 2.0 * cx * T_inlet
                else:
                    rows.append(k); cols.append(idx(i, j - 1)); vals.append(-2.0 * cx)
            else:
                if j - 1 == 0:
                    b[k] += cx * T_inlet
                else:
                    rows.append(k); cols.append(idx(i, j - 1)); vals.append(-cx)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-cx)

            # --- Y 方向扩散 ---
            if i == 0:
                # 下边界 Neumann：T_{-1} = T_{1}
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-2.0 * cy)
            elif i == ny - 1:
                # 上边界 Neumann：T_{ny} = T_{ny-2}
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-2.0 * cy)
            else:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-cy)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-cy)

            # --- X 方向对流（一阶迎风，u >= 0）---
            # rho*Cp*u * (T_j - T_{j-1}) / dx
            adv = rho_Cp * u[i] / dx
            rows.append(k); cols.append(k); vals.append(adv)
            if j - 1 == 0:
                b[k] += adv * T_inlet
            else:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-adv)

    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    T_vec = spsolve(A, b)
    T = T_vec.reshape((ny, nx))

    return X, Y, T, u


def test_solve_2d_phi_2_T(nx=201, ny=101):
    """
    用一个简单的高斯分布 phi 测试 solve_2d_phi_2_T，并画出温度场。
    """
    L = physics_params['L']
    H = physics_params['H']

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    # 构造一个测试用的 phi：中心在 (1.0, 0.5) 的高斯峰
    phi = np.exp(-((X - 1.2) ** 2 + (Y - 0.36) ** 2) / 0.05) * 1e10 + np.exp(-((X - 0.273) ** 2 + (Y - 0.652) ** 2) / 0.05) * 5e9

    # 提示：当前物理参数下 epsilon_f*Sigma_f 很小，真实 phi 量级较大时温升才明显。
    # 为了可视化，可临时放大 epsilon_f，例如：physics_params['epsilon_f'] *= 1e8

    # 求解温度场
    X, Y, T, u = solve_2d_phi_2_T(phi, nx=nx, ny=ny, T_inlet=600.0)
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(phi, extent=[x_min, x_max, y_min, y_max],
                         origin='lower', cmap='viridis', interpolation='bicubic', aspect='auto')
    axes[0].set_title(r'Neutron flux $\phi$')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(T, extent=[x_min, x_max, y_min, y_max],
                         origin='lower', cmap='viridis', interpolation='bicubic', aspect='auto')
    axes[1].set_title(r'Temperature $T$ [K]')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    fig.colorbar(im1, ax=axes[1])

    # 中心线温度分布
    mid_j = nx // 2
    axes[2].plot(y, T[:, mid_j], 'b-', linewidth=2, label=f'x = {x[mid_j]:.2f} m')
    axes[2].axhline(600.0, color='gray', linestyle='--', label='inlet temp')
    axes[2].set_xlabel('y [m]')
    axes[2].set_ylabel('T [K]')
    axes[2].set_title('Center-line temperature')
    axes[2].legend()

    plt.tight_layout()
    # plt.savefig('test_solve_2d_phi_2_T.png', dpi=300)
    plt.show()

    print(f"T min = {T.min():.4f} K")
    print(f"T max = {T.max():.4f} K")
    print(f"T at outlet center = {T[ny // 2, -1]:.4f} K")


def solve_2d_phi_2_C(phi, nx=201, ny=101, C_inlet=0.0):
    """
    求解二维稳态单组缓发中子先驱核输运方程:
        D_c * (d2C/dx2 + d2C/dy2) - u(y) * dC/dx - lambda * C + beta * nu * Sigma_f * phi = 0

    默认边界条件：
        - 左边界 (x=0): Dirichlet，C = C_inlet
        - 右边界 (x=L): Neumann，dC/dx = 0
        - 下边界 (y=0): Neumann，dC/dy = 0
        - 上边界 (y=H): Neumann，dC/dy = 0

    phi 可以是 2D numpy 数组 (ny, nx) 或 callable(x_vec, y_vec)。
    返回 X, Y, C 三个 2D 数组，以及 u 速度剖面 1D 数组。
    """
    L = physics_params['L']
    H = physics_params['H']
    U_max = physics_params['U_max']
    D_c = physics_params['D_c']
    beta = physics_params['beta']
    nu = physics_params['nu']
    Sigma_f = physics_params['Sigma_f']
    lam = physics_params['lambda']

    dx = L / (nx - 1)
    dy = H / (ny - 1)

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    # Poiseuille 速度剖面，u >= 0
    u = U_max * 4.0 * (y / H) * (1.0 - y / H)

    # 构造先驱核产生项
    if callable(phi):
        phi_vals = phi(X, Y)
    else:
        if phi.shape != (ny, nx):
            raise ValueError(f"phi shape {phi.shape} 与网格 (ny={ny}, nx={nx}) 不一致")
        phi_vals = phi
    source = beta * nu * Sigma_f * phi_vals

    N = nx * ny

    def idx(i, j):
        return i * nx + j

    cx = D_c / (dx ** 2)
    cy = D_c / (dy ** 2)

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)

            # 左边界 Dirichlet
            if j == 0:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = C_inlet
                continue

            # 右端源项
            b[k] = source[i, j]

            # 扩散 Laplacian 对角项
            rows.append(k); cols.append(k); vals.append(2.0 * cx + 2.0 * cy + lam)

            # --- X 方向扩散 ---
            if j == nx - 1:
                # 右边界 Neumann：C_{j+1} = C_{j-1}
                if j - 1 == 0:
                    b[k] += 2.0 * cx * C_inlet
                else:
                    rows.append(k); cols.append(idx(i, j - 1)); vals.append(-2.0 * cx)
            else:
                if j - 1 == 0:
                    b[k] += cx * C_inlet
                else:
                    rows.append(k); cols.append(idx(i, j - 1)); vals.append(-cx)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-cx)

            # --- Y 方向扩散 ---
            if i == 0:
                # 下边界 Neumann：C_{-1} = C_{1}
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-2.0 * cy)
            elif i == ny - 1:
                # 上边界 Neumann：C_{ny} = C_{ny-2}
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-2.0 * cy)
            else:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-cy)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-cy)

            # --- X 方向对流（一阶迎风，u >= 0）---
            # u(y) * (C_j - C_{j-1}) / dx
            adv = u[i] / dx
            rows.append(k); cols.append(k); vals.append(adv)
            if j - 1 == 0:
                b[k] += adv * C_inlet
            else:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-adv)

    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    C_vec = spsolve(A, b)
    C = C_vec.reshape((ny, nx))

    return X, Y, C, u


def test_solve_2d_phi_2_C(nx=201, ny=101):
    """
    用一个简单的高斯分布 phi 测试 solve_2d_phi_2_C，并画出先驱核浓度场。
    """
    L = physics_params['L']
    H = physics_params['H']

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    # 构造一个测试用的 phi：中心在 (1.0, 0.5) 的高斯峰
    phi = np.exp(-((X - 1.2) ** 2 + (Y - 0.36) ** 2) / 0.05) * 1e10 + np.exp(-((X - 0.273) ** 2 + (Y - 0.652) ** 2) / 0.05) * 5e9

    # 求解先驱核浓度场
    X, Y, C, u = solve_2d_phi_2_C(phi, nx=nx, ny=ny, C_inlet=0.0)
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(phi, extent=[x_min, x_max, y_min, y_max],
                         origin='lower', cmap='viridis', interpolation='bicubic', aspect='auto')
    axes[0].set_title(r'Neutron flux $\phi$')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(C, extent=[x_min, x_max, y_min, y_max],
                         origin='lower', cmap='viridis', interpolation='bicubic', aspect='auto')
    axes[1].set_title(r'Precursor concentration $C$')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    fig.colorbar(im1, ax=axes[1])

    # 中心线浓度分布
    mid_j = nx // 2
    axes[2].plot(y, C[:, mid_j], 'b-', linewidth=2, label=f'x = {x[mid_j]:.2f} m')
    axes[2].axhline(0.0, color='gray', linestyle='--', label='inlet conc')
    axes[2].set_xlabel('y [m]')
    axes[2].set_ylabel('C')
    axes[2].set_title('Center-line precursor concentration')
    axes[2].legend()

    plt.tight_layout()
    # plt.savefig('test_solve_2d_phi_2_C.png', dpi=300)
    plt.show()

    print(f"C min = {C.min():.4e}")
    print(f"C max = {C.max():.4e}")
    print(f"C at outlet center = {C[ny // 2, -1]:.4e}")


def solve_2d_T_C_2_phi(T, C, nx=201, ny=101):
    """
    求解二维稳态单能群中子扩散方程（固定源问题）:
        D*(phi_xx + phi_yy) - Sigma_a(T)*phi + (1-beta)*nu*Sigma_f*phi + lambda*C + S_ext = 0

    温度反馈吸收截面:
        Sigma_a(T) = Sigma_a0 * [1 + alpha_T * (T - T0)]

    边界条件：四周真空边界，phi = 0。

    T, C 为 2D numpy 数组 (ny, nx)，分别表示温度场 [K] 和先驱核浓度场。
    返回 X, Y, phi 三个 2D 数组。
    """
    L = physics_params['L']
    H = physics_params['H']
    D = physics_params['D']
    Sigma_a0 = physics_params['Sigma_a0']
    beta = physics_params['beta']
    nu_Sigma_f = physics_params['nu_Sigma_f']
    lam = physics_params['lambda']
    T0 = physics_params['T0']
    alpha_T = physics_params['alpha_T']
    S0 = physics_params['S0']
    x0 = physics_params['x0']
    y0 = physics_params['y0']
    sigma_x = physics_params['sigma_x']
    sigma_y = physics_params['sigma_y']

    dx = L / (nx - 1)
    dy = H / (ny - 1)

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    if T.shape != (ny, nx) or C.shape != (ny, nx):
        raise ValueError(f"T shape {T.shape} or C shape {C.shape} 与网格 (ny={ny}, nx={nx}) 不一致")

    # 温度反馈吸收截面
    Sigma_a = Sigma_a0 * (1.0 + alpha_T * (T - T0))

    # 外中子源（高斯分布）
    S_ext = S0 * np.exp(-((X - x0) ** 2) / (2.0 * sigma_x ** 2)
                        - ((Y - y0) ** 2) / (2.0 * sigma_y ** 2))

    # 有效反应性去除截面（移到右端后的系数）
    removal = Sigma_a - (1.0 - beta) * nu_Sigma_f

    N = nx * ny

    def idx(i, j):
        return i * nx + j

    cx = D / (dx ** 2)
    cy = D / (dy ** 2)

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)

            # 四周真空边界：phi = 0
            if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = 0.0
                continue

            # 右端源项: lambda*C + S_ext
            b[k] = lam * C[i, j] + S_ext[i, j]

            # 扩散 Laplacian 对角项（含反应性去除项）
            # -D*Laplacian(phi) + removal*phi = RHS
            rows.append(k); cols.append(k); vals.append(2.0 * cx + 2.0 * cy + removal[i, j])

            # X 方向扩散
            rows.append(k); cols.append(idx(i, j - 1)); vals.append(-cx)
            rows.append(k); cols.append(idx(i, j + 1)); vals.append(-cx)

            # Y 方向扩散
            rows.append(k); cols.append(idx(i - 1, j)); vals.append(-cy)
            rows.append(k); cols.append(idx(i + 1, j)); vals.append(-cy)

    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    phi_vec = spsolve(A, b)
    phi = phi_vec.reshape((ny, nx))

    return X, Y, phi


def test_solve_2d_T_C_2_phi(nx=201, ny=101):
    """
    读取 1_FFT_T_10000_101_201.npy 和 1_FFT_C_10000_101_201.npy 的第一个样本，
    用 solve_2d_T_C_2_phi 求解 phi，并与 1_FFT_phi_10000_201_101.npy 的第一个真值 phi 对比可视化。
    画出五张图：T、C、S_ext、求解的 phi、真值 phi。
    """
    L = physics_params['L']
    H = physics_params['H']

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # 读取第一个 T 和 C 样本
    T = np.load('1_FFT_T_10000_101_201.npy')[0]
    C = np.load('1_FFT_C_10000_101_201.npy')[0]

    # 求解 phi
    X, Y, phi = solve_2d_T_C_2_phi(T, C, nx=nx, ny=ny)

    # 读取第一个真值 phi（注意 shape 为 (201, 101)，需转置为 (ny, nx)）
    phi_true = np.load('1_FFT_phi_10000_201_101.npy')[0].T

    # 构造与求解器相同的固定外源 S_ext
    S0 = physics_params['S0']
    x0 = physics_params['x0']
    y0 = physics_params['y0']
    sigma_x = physics_params['sigma_x']
    sigma_y = physics_params['sigma_y']
    S_ext = S0 * np.exp(-((X - x0) ** 2) / (2.0 * sigma_x ** 2)
                        - ((Y - y0) ** 2) / (2.0 * sigma_y ** 2))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axes[0, 0].imshow(T, extent=[x_min, x_max, y_min, y_max],
                            origin='lower', cmap='hot', interpolation='bicubic', aspect='auto')
    axes[0, 0].set_title(r'Temperature $T$ [K] (1_FFT_T sample 0)')
    axes[0, 0].set_xlabel('x [m]')
    axes[0, 0].set_ylabel('y [m]')
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(C, extent=[x_min, x_max, y_min, y_max],
                            origin='lower', cmap='viridis', interpolation='bicubic', aspect='auto')
    axes[0, 1].set_title(r'Precursor concentration $C$ (1_FFT_C sample 0)')
    axes[0, 1].set_xlabel('x [m]')
    axes[0, 1].set_ylabel('y [m]')
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(S_ext, extent=[x_min, x_max, y_min, y_max],
                            origin='lower', cmap='inferno', interpolation='bicubic', aspect='auto')
    axes[0, 2].set_title(r'External source $S_{\mathrm{ext}}$')
    axes[0, 2].set_xlabel('x [m]')
    axes[0, 2].set_ylabel('y [m]')
    fig.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(phi, extent=[x_min, x_max, y_min, y_max],
                            origin='lower', cmap='plasma', interpolation='bicubic', aspect='auto')
    axes[1, 0].set_title(r'Computed neutron flux $\phi$')
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('y [m]')
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(phi_true, extent=[x_min, x_max, y_min, y_max],
                            origin='lower', cmap='plasma', interpolation='bicubic', aspect='auto')
    axes[1, 1].set_title(r'True neutron flux $\phi$ (1_FFT_phi sample 0)')
    axes[1, 1].set_xlabel('x [m]')
    axes[1, 1].set_ylabel('y [m]')
    fig.colorbar(im4, ax=axes[1, 1])

    # 关闭多余的子图
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('test_solve_2d_T_C_2_phi_sample0.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"computed phi min = {phi.min():.4e}, max = {phi.max():.4e}, mean = {phi.mean():.4e}")
    print(f"true     phi min = {phi_true.min():.4e}, max = {phi_true.max():.4e}, mean = {phi_true.mean():.4e}")


def generate_phi_2d_grf(n_samples, nx, ny, Lx=2.0, Ly=1.0,
                        rel_l_macro=(0.25, 0.45),  # 宏观相对比例
                        rel_l_micro=(0.08, 0.18),  # 微观相对比例
                        sigma_macro=1.0,
                        sigma_micro=0.15,          # 微观标准差 (~15% 涨落)
                        phi_peak_range=(1.0e7, 2.5e7),  # 提高上限以增大 phi_max 覆盖
                        max_laplacian_threshold=None,     # Laplacian 过滤阈值 (None 表示不过滤)
                        seed=None):
    """
    完备版 2D GRF 中子通量场生成器：
    1. 边界/正定: grf_macro 满足四周 0 边界且 >=0，grf_micro 通过 exp() 保证正定；
    2. 物理量级: 自动归一化并随机采样物理峰值 (如 1e13 ~ 5e13)；
    3. 梯度/二阶导过滤: 支持 max_laplacian_threshold，自动剔除局部过陡/难收敛的样本。
    """
    rng = np.random.default_rng(seed)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # 1. 构建物理空间网格与边界窗函数 (Sine Window)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    boundary_window = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

    # 计算内部区域遮罩 (避免边界归零区域对 Laplacian 检查产生误判)
    Xn, Yn = X / Lx, Y / Ly
    x_cut, y_cut = 0.02, 0.02
    interior = (Xn >= x_cut) & (Xn <= 1.0 - x_cut) & (Yn >= y_cut) & (Yn <= 1.0 - y_cut)

    # 2. 2D 频域网格
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    def _sample_pure_grf_2d(l_x, l_y, sigma_amp):
        psd_2d = (sigma_amp ** 2 * 2 * np.pi * l_x * l_y) * \
                 np.exp(-2 * np.pi ** 2 * (KX ** 2 * l_x ** 2 + KY ** 2 * l_y ** 2))

        white_noise = rng.standard_normal((nx, ny))
        noise_fft = np.fft.fft2(white_noise)
        amplitude = np.sqrt(psd_2d / (dx * dy))
        return np.fft.ifft2(noise_fft * amplitude).real

    phi_samples, phi_baselines, grf_micros = [], [], []
    rejected = 0

    for _ in range(n_samples):
        # 尝试生成满足 Laplacian 阈值约束的样本 (最多尝试 100 次)
        for _attempt in range(100):
            phi_peak = rng.uniform(*phi_peak_range)

            # ------------------- 1. 生成宏观基线 (Phi_base) -------------------
            l_x_macro = rng.uniform(*rel_l_macro) * Lx
            l_y_macro = rng.uniform(*rel_l_macro) * Ly
            raw_macro = _sample_pure_grf_2d(l_x_macro, l_y_macro, sigma_macro)

            macro_positive = raw_macro - np.min(raw_macro)
            raw_base = macro_positive * boundary_window
            Phi_base = phi_peak * (raw_base / (np.max(raw_base) + 1e-12))

            # ------------------- 2. 生成微观场 (grf_micro) -------------------
            l_x_micro = rng.uniform(*rel_l_micro) * Lx
            l_y_micro = rng.uniform(*rel_l_micro) * Ly

            raw_micro = _sample_pure_grf_2d(l_x_micro, l_y_micro, sigma_micro)
            grf_micro = np.exp(raw_micro)  # 严格正定无量纲纹理

            # ------------------- 3. 合成与 Laplacian 检查 -------------------
            Phi_final = Phi_base * grf_micro

            # 梯度/二阶导数过大过滤逻辑
            if max_laplacian_threshold is not None:
                # 二阶中心差分计算 Laplacian
                phi_xx = np.gradient(np.gradient(Phi_final, dx, axis=0), dx, axis=0)
                phi_yy = np.gradient(np.gradient(Phi_final, dy, axis=1), dy, axis=1)
                laplacian = np.abs(phi_xx + phi_yy)

                # 仅检查内部区域，若超过设定的最大二阶导阈值，放弃当前样本重抽
                if np.max(laplacian[interior]) > max_laplacian_threshold:
                    rejected += 1
                    continue

            # 检查通过，保存样本
            phi_samples.append(Phi_final)
            phi_baselines.append(Phi_base)
            grf_micros.append(grf_micro)
            break

    if rejected > 0:
        total = len(phi_samples) + rejected
        print(f"generate_phi_2d_grf: 过滤剔除 {rejected}/{total} 个梯度过大样本 "
              f"({rejected / total:.1%}), 成功保留 {len(phi_samples)} 个")

    return np.array(phi_samples), np.array(phi_baselines), np.array(grf_micros)


def generate_T_2d_grf(n_samples, nx, ny, Lx=2.0, Ly=1.0,
                      T_inlet=600.0,
                      T_amp_range=(150.0, 600.0),  # 提高下限以增大均值，匹配耦合结果 T_max~1061 K
                      x_freq_range=(0.95, 1.05),
                      y_freq_range=(0.50, 2.00),
                      y_amp_range=(0.05, 0.30),
                      y_phase_range=(0.0, 2.0 * np.pi),
                      rel_l_micro=(0.08, 0.18),
                      sigma_micro=0.06,
                      seed=None):
    """
    生成温度场 T(x, y), 输出 shape 为 (n_samples, ny, nx).

    设计要点:
        - 宏观基线为 x 方向 cos 上升线 (1 - cos(π * freq_x * x / Lx))/2;
        - 叠加 y 方向 sin/cos 波动，形成 T_true 中的弯曲条带;
        - 微观纹理用 exp(FFT-GRF) 提供高频扰动;
        - 全场 T >= T_inlet, 左边界 Dirichlet, 其余三边 Neumann。
    """
    rng = np.random.default_rng(seed)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Xn = X / Lx
    Yn = Y / Ly

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    def _sample_pure_grf_2d(l_x, l_y, sigma_amp):
        psd_2d = (sigma_amp ** 2 * 2 * np.pi * l_x * l_y) *                  np.exp(-2 * np.pi ** 2 * (KX ** 2 * l_x ** 2 + KY ** 2 * l_y ** 2))
        white_noise = rng.standard_normal((nx, ny))
        noise_fft = np.fft.fft2(white_noise)
        amplitude = np.sqrt(psd_2d / (dx * dy))
        return np.fft.ifft2(noise_fft * amplitude).real

    T_samples = []

    for _ in range(n_samples):
        # 99.5% 概率从原区间采样，0.5% 概率从高尾区间采样，以覆盖真实数据中的极端峰值
        if rng.random() < 0.005:
            T_amp = rng.uniform(900.0, 1100.0)
        else:
            T_amp = rng.uniform(*T_amp_range)
        x_freq = rng.uniform(*x_freq_range)
        y_freq = rng.uniform(*y_freq_range)
        y_amp = rng.uniform(*y_amp_range)
        y_phase = rng.uniform(*y_phase_range)

        # x 方向单调上升基线，x=0 处为 0，x=Lx 处为 1
        x_base = 0.5 * (1.0 - np.cos(np.pi * x_freq * Xn))

        # y 方向波动
        y_ripple = 1.0 + y_amp * np.cos(np.pi * y_freq * Yn + y_phase)

        # 宏观基线
        T_base = T_amp * x_base * y_ripple

        # 微观正定纹理
        l_x_micro = rng.uniform(*rel_l_micro) * Lx
        l_y_micro = rng.uniform(*rel_l_micro) * Ly
        raw_micro = _sample_pure_grf_2d(l_x_micro, l_y_micro, sigma_micro)
        T_perturb = np.exp(raw_micro)

        # 合成
        T_final = T_inlet + T_base * T_perturb

        # 边界条件
        T_final[0, :] = T_inlet

        # 右 / 下 / 上边界零法向梯度 (Neumann), 三点格式
        T_final[-1, :] = (4.0 * T_final[-2, :] - T_final[-3, :]) / 3.0
        T_final[:, 0] = (4.0 * T_final[:, 1] - T_final[:, 2]) / 3.0
        T_final[:, -1] = (4.0 * T_final[:, -2] - T_final[:, -3]) / 3.0

        T_samples.append(T_final.T)

    return np.array(T_samples)


def generate_phi_2d_grf_raw(n_samples, nx, ny, Lx=2.0, Ly=1.0,
                            rel_l_macro=(0.2, 0.4),  # 相对物理尺寸 Lx, Ly 的比例 (0.2-0.4 能看到完美的二维起伏)
                            rel_l_micro=(0.05, 0.12),  # 微观相对比例
                            sigma_macro=1.0,
                            sigma_micro=0.3,
                            seed=None):
    """
    修正后的纯净 2D-FFT GRF (彻底解决 y 方向与 x 方向尺度不对称导致的 y 方向平铺问题)
    """
    rng = np.random.default_rng(seed)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # 2D 频域网格 (正确匹配 2D FFT 空间频率)
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    def _sample_pure_grf_2d(l_x, l_y, sigma_amp):
        # 2D 物理相关功率谱 density
        psd_2d = (sigma_amp ** 2 * 2 * np.pi * l_x * l_y) * \
                 np.exp(-2 * np.pi ** 2 * (KX ** 2 * l_x ** 2 + KY ** 2 * l_y ** 2))

        white_noise = rng.standard_normal((nx, ny))
        noise_fft = np.fft.fft2(white_noise)
        amplitude = np.sqrt(psd_2d / (dx * dy))
        return np.fft.ifft2(noise_fft * amplitude).real

    raw_samples, grf_macros, grf_micros = [], [], []

    for _ in range(n_samples):
        # 核心修复: 按物理区域尺寸的相对比例随机抽样，保证 x 和 y 方向的波形特征各向同性！
        alpha_x_macro = rng.uniform(*rel_l_macro)
        alpha_y_macro = rng.uniform(*rel_l_macro)
        l_x_macro = alpha_x_macro * Lx
        l_y_macro = alpha_y_macro * Ly

        grf_macro = _sample_pure_grf_2d(l_x_macro, l_y_macro, sigma_macro)

        alpha_x_micro = rng.uniform(*rel_l_micro)
        alpha_y_micro = rng.uniform(*rel_l_micro)
        l_x_micro = alpha_x_micro * Lx
        l_y_micro = alpha_y_micro * Ly

        grf_micro = _sample_pure_grf_2d(l_x_micro, l_y_micro, sigma_micro)

        raw_final = grf_macro * grf_micro

        raw_samples.append(raw_final)
        grf_macros.append(grf_macro)
        grf_micros.append(grf_micro)

    return np.array(raw_samples), np.array(grf_macros), np.array(grf_micros)


def visualize_phi_2d_samples(samples, nx, ny, Lx=2.0, Ly=1.0, save_path='phi_samples_visualization.png'):
    """
    可视化生成的 phi 样本 (类似 visualize_T_2d_samples 的多样本视图).
    默认绘制前 10 个样本的 2D 中子通量分布.
    """
    n_plot = min(10, len(samples))
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    nrows = 2
    ncols = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_plot):
        ax = axes[i]
        phi = samples[i]  # (nx, ny)
        vmin, vmax = phi.min(), phi.max()
        c = ax.contourf(X, Y, phi, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Sample {i}: phi={vmin:.2e}~{vmax:.2e}')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')


def generate_phi_10000_samples():
    # 生成 10000 个 2D GRF 中子通量场样本
    n_samples = 10000
    nx, ny = 201, 101

    samples, baselines, grf_factors = generate_phi_2d_grf(
        n_samples=n_samples,
        nx=nx, ny=ny,
        Lx=2.0, Ly=1.0,
        rel_l_macro=(0.15, 0.60),   # 更宽宏观尺度 -> 更多样的基线形态
        rel_l_micro=(0.05, 0.25),   # 更宽微观尺度 -> 更丰富的局部纹理
        sigma_macro=1.0,
        sigma_micro=0.25,           # 稍强微观扰动 -> 样本间差异更大
        max_laplacian_threshold=None,
    )

    visualize_phi_2d_samples(samples, nx, ny)

    np.save('1_FFT_phi_10000_201_101.npy', samples)
    print("Saved 1_FFT_phi_10000_201_101.npy")


def visualize_T_2d_samples(samples, nx, ny, Lx=2.0, Ly=1.0, save_path='T_samples_visualization.png'):
    """
    可视化生成的 T 样本 (类似 visualize_phi_2d_samples 的多样本视图).
    默认绘制前 10 个样本的 2D 温度分布.
    """
    n_plot = min(10, len(samples))
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    nrows = 2
    ncols = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_plot):
        ax = axes[i]
        T = samples[i].T  # (nx, ny)
        vmin, vmax = T.min(), T.max()
        c = ax.contourf(X, Y, T, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Sample {i}: T={vmin:.0f}~{vmax:.0f} K')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')


def generate_T_10000_samples():
    """
    生成 10000 个满足 T_C_2_phi 输入约束的温度场 T 样本,
    并保存为 1_FFT_T_10000_101_201.npy (shape: 10000, ny, nx).
    """
    n_samples = 10000
    nx, ny = 201, 101

    T_samples = generate_T_2d_grf(
        n_samples=n_samples,
        nx=nx, ny=ny,
        Lx=2.0, Ly=1.0,
        T_inlet=600.0,
        x_freq_range=(0.95, 1.05),
        y_freq_range=(0.50, 2.00),
        y_amp_range=(0.05, 0.30),
        y_phase_range=(0.0, 2.0 * np.pi),
        rel_l_micro=(0.08, 0.18),
        sigma_micro=0.06,
        seed=None,
    )

    print(f"T_samples shape: {T_samples.shape}")
    print(f"T min = {T_samples.min():.2f}, max = {T_samples.max():.2f}")

    visualize_T_2d_samples(T_samples, nx, ny)

    np.save('1_FFT_T_10000_101_201.npy', T_samples)
    print("Saved 1_FFT_T_10000_101_201.npy")


def generate_C_2d_grf(n_samples, nx, ny, Lx=2.0, Ly=1.0,
                      C_inlet=0.0,
                      C_amp_range=(4.0e5, 8.0e5),  # 提高下限以进一步增大 C 均值
                      x_peak_range=(1.00, 1.80),
                      y_peak_range=(0.25, 0.75),
                      sigma_x_range=(0.40, 1.00),
                      sigma_y_range=(0.40, 1.20),
                      x_rise_range=(1.30, 1.40),
                      x_cos_window=False,
                      x_window_strength_range=(0.30, 1.00),
                      rel_l_micro=(0.08, 0.18),
                      sigma_micro=0.08,
                      seed=None):
    """
    生成先驱核浓度场 C(x, y), 输出 shape 为 (n_samples, ny, nx).

    设计要点:
        - 宏观基线为随机高斯斑点，峰值位置、振幅、x/y 扩散宽度均随机;
        - 左边界用 (1 - exp(-x/x_rise)) 保证 C(0,y)≈0;
        - 可选 x 方向 cos 窗，强度随机 (0 表示无窗, 1 表示完整 cos 窗);
        - 微观纹理用 exp(FFT-GRF) 提供高频扰动;
        - 左边界 Dirichlet, 其余三边 Neumann, 全场 C >= C_inlet。
    """
    rng = np.random.default_rng(seed)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    def _sample_pure_grf_2d(l_x, l_y, sigma_amp):
        psd_2d = (sigma_amp ** 2 * 2 * np.pi * l_x * l_y) * \
                 np.exp(-2 * np.pi ** 2 * (KX ** 2 * l_x ** 2 + KY ** 2 * l_y ** 2))
        white_noise = rng.standard_normal((nx, ny))
        noise_fft = np.fft.fft2(white_noise)
        amplitude = np.sqrt(psd_2d / (dx * dy))
        return np.fft.ifft2(noise_fft * amplitude).real

    C_samples = []

    for _ in range(n_samples):
        # 99.5% 概率从原区间采样，0.5% 概率从高尾区间采样，以覆盖真实数据中的极端峰值
        if rng.random() < 0.005:
            C_amp = rng.uniform(8.0e5, 1.0e6)
        else:
            C_amp = rng.uniform(*C_amp_range)
        x_peak = rng.uniform(*x_peak_range)
        y_peak = rng.uniform(*y_peak_range)
        sigma_x = rng.uniform(*sigma_x_range)
        sigma_y = rng.uniform(*sigma_y_range)
        x_rise = rng.uniform(*x_rise_range)

        # 宏观高斯斑点，乘左边界上升包络和可选随机强度 x 方向 cos 窗
        gaussian = np.exp(-((X - x_peak) ** 2) / (2.0 * sigma_x ** 2)
                          - ((Y - y_peak) ** 2) / (2.0 * sigma_y ** 2))
        left_env = 1.0 - np.exp(-X / x_rise)
        if x_cos_window:
            x_window_strength = rng.uniform(*x_window_strength_range)
            x_window = 1.0 - x_window_strength * 0.5 * (1.0 - np.cos(np.pi * X / Lx))
        else:
            x_window = 1.0
        C_base = C_amp * gaussian * left_env * x_window

        # 微观正定纹理
        l_x_micro = rng.uniform(*rel_l_micro) * Lx
        l_y_micro = rng.uniform(*rel_l_micro) * Ly
        raw_micro = _sample_pure_grf_2d(l_x_micro, l_y_micro, sigma_micro)
        C_perturb = np.exp(raw_micro)

        # 合成
        C_final = C_inlet + C_base * C_perturb

        # 边界条件
        C_final[0, :] = C_inlet
        C_final = np.maximum(C_final, C_inlet)

        # 右 / 下 / 上边界零法向梯度 (Neumann), 三点格式
        C_final[-1, :] = (4.0 * C_final[-2, :] - C_final[-3, :]) / 3.0
        C_final[:, 0] = (4.0 * C_final[:, 1] - C_final[:, 2]) / 3.0
        C_final[:, -1] = (4.0 * C_final[:, -2] - C_final[:, -3]) / 3.0

        C_final = np.maximum(C_final, C_inlet)

        C_samples.append(C_final.T)

    return np.array(C_samples)


def visualize_C_2d_samples(samples, nx, ny, Lx=2.0, Ly=1.0, save_path='C_samples_visualization.png'):
    """
    可视化生成的 C 样本.
    默认绘制前 10 个样本的 2D 先驱核浓度分布.
    """
    n_plot = min(10, len(samples))
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    nrows = 2
    ncols = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_plot):
        ax = axes[i]
        C = samples[i].T  # (nx, ny)
        vmin, vmax = C.min(), C.max()
        c = ax.contourf(X, Y, C, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Sample {i}: C={vmin:.2e}~{vmax:.2e}')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')


def generate_C_10000_samples():
    """
    生成 10000 个满足 T_C_2_phi 输入约束的先驱核浓度场 C 样本,
    并保存为 1_FFT_C_10000_101_201.npy (shape: 10000, ny, nx).
    """
    n_samples = 10000
    nx, ny = 201, 101

    C_samples = generate_C_2d_grf(
        n_samples=n_samples,
        nx=nx, ny=ny,
        Lx=2.0, Ly=1.0,
        C_inlet=0.0,
        x_peak_range=(1.00, 1.80),
        y_peak_range=(0.25, 0.75),
        sigma_x_range=(0.40, 1.00),
        sigma_y_range=(0.40, 1.20),
        x_rise_range=(1.30, 1.40),
        x_cos_window=False,
        x_window_strength_range=(0.30, 1.00),
        rel_l_micro=(0.08, 0.18),
        sigma_micro=0.08,
        seed=None,
    )

    print(f"C_samples shape: {C_samples.shape}")
    print(f"C min = {C_samples.min():.2e}, max = {C_samples.max():.2e}")

    visualize_C_2d_samples(C_samples, nx, ny)

    np.save('1_FFT_C_10000_101_201.npy', C_samples)
    print("Saved 1_FFT_C_10000_101_201.npy")

def generate_true_solutions(nx=201, ny=101, n_samples=10000,
                            phi_path='1_FFT_phi_10000_201_101.npy',
                            T_path='1_FFT_T_10000_101_201.npy',
                            C_path='1_FFT_C_10000_101_201.npy',
                            save_T_true='T_true_10000_101_201.npy',
                            save_C_true='C_true_10000_101_201.npy',
                            save_phi_true='phi_true_10000_201_101.npy',
                            progress_interval=100):
    """
    根据已生成的 FFT-GRF 样本，调用数值求解器生成真值 (ground-truth) 场：
        - T_true: 以 phi 为输入，用 solve_2d_phi_2_T 求解得到，shape (n_samples, ny, nx)
        - C_true: 以 phi 为输入，用 solve_2d_phi_2_C 求解得到，shape (n_samples, ny, nx)
        - phi_true: 以 T、C 为输入，用 solve_2d_T_C_2_phi 求解得到，shape (n_samples, nx, ny)

    参数:
        nx, ny              : 网格数 (默认 201, 101)
        n_samples           : 样本数 (默认 10000)
        phi_path            : phi FFT 样本路径，shape (n_samples, nx, ny)
        T_path              : T FFT 样本路径，shape (n_samples, ny, nx)
        C_path              : C FFT 样本路径，shape (n_samples, ny, nx)
        save_T_true         : 保存 T_true 的文件名
        save_C_true         : 保存 C_true 的文件名
        save_phi_true       : 保存 phi_true 的文件名
        progress_interval   : 进度打印间隔
    """
    print("Loading FFT-GRF samples...")
    phi_samples = np.load(phi_path)[:n_samples]   # (n_samples, nx, ny)
    T_samples = np.load(T_path)[:n_samples]       # (n_samples, ny, nx)
    C_samples = np.load(C_path)[:n_samples]       # (n_samples, ny, nx)

    assert phi_samples.shape == (n_samples, nx, ny), f"phi shape {phi_samples.shape} != ({n_samples}, {nx}, {ny})"
    assert T_samples.shape == (n_samples, ny, nx), f"T shape {T_samples.shape} != ({n_samples}, {ny}, {nx})"
    assert C_samples.shape == (n_samples, ny, nx), f"C shape {C_samples.shape} != ({n_samples}, {ny}, {nx})"

    T_true = np.zeros((n_samples, ny, nx), dtype=np.float64)
    C_true = np.zeros((n_samples, ny, nx), dtype=np.float64)
    phi_true = np.zeros((n_samples, nx, ny), dtype=np.float64)

    print("Generating T_true and C_true from phi_samples...")
    for i in range(n_samples):
        phi_i = phi_samples[i].T  # solver 需要 (ny, nx)
        _, _, T_i, _ = solve_2d_phi_2_T(phi_i, nx=nx, ny=ny)
        _, _, C_i, _ = solve_2d_phi_2_C(phi_i, nx=nx, ny=ny)
        T_true[i] = T_i
        C_true[i] = C_i
        if (i + 1) % progress_interval == 0:
            print(f"  phi->T/C: {i + 1}/{n_samples} done")

    print("T_true/C_true statistics:")
    print(f"  T: min={T_true.min():.2f}, max={T_true.max():.2f}, mean={T_true.mean():.2f}")
    print(f"  C: min={C_true.min():.4e}, max={C_true.max():.4e}, mean={C_true.mean():.4e}")

    np.save(save_T_true, T_true)
    np.save(save_C_true, C_true)
    print(f"Saved {save_T_true} and {save_C_true}")

    print("Generating phi_true from T_samples and C_samples...")
    for i in range(n_samples):
        T_i = T_samples[i]        # (ny, nx)
        C_i = C_samples[i]        # (ny, nx)
        _, _, phi_i = solve_2d_T_C_2_phi(T_i, C_i, nx=nx, ny=ny)
        phi_true[i] = phi_i.T     # 保存为 (nx, ny) 以匹配 phi 样本维度
        if (i + 1) % progress_interval == 0:
            print(f"  T,C->phi: {i + 1}/{n_samples} done")

    print("phi_true statistics:")
    print(f"  phi: min={phi_true.min():.4e}, max={phi_true.max():.4e}, mean={phi_true.mean():.4e}")

    np.save(save_phi_true, phi_true)
    print(f"Saved {save_phi_true}")

    return T_true, C_true, phi_true


def solve_2d_coupled_phi_T_C(nx=201, ny=101, max_iter=100, tol=1e-6, relax=0.5, verbose=True):
    """
    迭代耦合求解 phi-T-C 稳态场：
        phi -> solve_2d_phi_2_T -> T
        phi -> solve_2d_phi_2_C -> C
        T, C -> solve_2d_T_C_2_phi -> phi_new

    使用松弛迭代，直到 T 场的相对变化小于 tol。

    返回:
        X, Y, phi, T, C, S_ext, T_errors
    """
    L = physics_params['L']
    H = physics_params['H']
    S0 = physics_params['S0']
    x0 = physics_params['x0']
    y0 = physics_params['y0']
    sigma_x = physics_params['sigma_x']
    sigma_y = physics_params['sigma_y']
    T0 = physics_params['T0']

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)

    # 构造固定外源 S_ext
    S_ext = S0 * np.exp(-((X - x0) ** 2) / (2.0 * sigma_x ** 2)
                        - ((Y - y0) ** 2) / (2.0 * sigma_y ** 2))

    # 初始化：冷态、无先驱核，求解初始 phi
    T = np.full((ny, nx), T0, dtype=np.float64)
    C = np.zeros((ny, nx), dtype=np.float64)
    _, _, phi = solve_2d_T_C_2_phi(T, C, nx=nx, ny=ny)

    T_errors = []

    for it in range(max_iter):
        # phi -> T, C
        _, _, T_new, _ = solve_2d_phi_2_T(phi, nx=nx, ny=ny, T_inlet=T0)
        _, _, C_new, _ = solve_2d_phi_2_C(phi, nx=nx, ny=ny, C_inlet=0.0)

        # T, C -> phi
        _, _, phi_new = solve_2d_T_C_2_phi(T_new, C_new, nx=nx, ny=ny)

        # 计算 T 的相对变化
        err_T = np.linalg.norm(T_new - T) / (np.linalg.norm(T) + 1e-12)
        T_errors.append(err_T)

        if verbose:
            print(f"iter {it:3d}: T_err = {err_T:.6e}, phi_max = {phi_new.max():.4e}, T_max = {T_new.max():.2f}")

        # 松弛更新
        phi = relax * phi_new + (1.0 - relax) * phi
        T = T_new
        C = C_new

        if err_T < tol:
            if verbose:
                print(f"Converged at iteration {it}")
            break
    else:
        if verbose:
            print(f"Reached max_iter = {max_iter}, final T_err = {err_T:.6e}")

    return X, Y, phi, T, C, S_ext, T_errors


def test_solve_2d_coupled_phi_T_C(nx=201, ny=101, max_iter=100, tol=1e-6, relax=0.5):
    """
    测试 solve_2d_coupled_phi_T_C，可视化：
        - T 收敛误差下降曲线
        - 最终 T、C、phi、S_ext 场
    """
    L = physics_params['L']
    H = physics_params['H']
    X, Y, phi, T, C, S_ext, T_errors = solve_2d_coupled_phi_T_C(
        nx=nx, ny=ny, max_iter=max_iter, tol=tol, relax=relax, verbose=True
    )

    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(x, y)
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 收敛曲线
    ax_err = fig.add_subplot(gs[0, :])
    ax_err.semilogy(T_errors, 'b-o', markersize=4)
    ax_err.set_xlabel('Iteration')
    ax_err.set_ylabel(r'Relative $||T_{new} - T_{old}||_2 / ||T||_2$')
    ax_err.set_title('Temperature convergence history')
    ax_err.grid(True, which='both', ls='--', alpha=0.5)

    # T
    ax_T = fig.add_subplot(gs[1, 0])
    im_T = ax_T.imshow(T, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='hot', interpolation='bicubic', aspect='auto')
    ax_T.set_title(r'Temperature $T$ [K]')
    ax_T.set_xlabel('x [m]')
    ax_T.set_ylabel('y [m]')
    fig.colorbar(im_T, ax=ax_T)

    # C
    ax_C = fig.add_subplot(gs[1, 1])
    im_C = ax_C.imshow(C, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='viridis', interpolation='bicubic', aspect='auto')
    ax_C.set_title(r'Precursor concentration $C$')
    ax_C.set_xlabel('x [m]')
    ax_C.set_ylabel('y [m]')
    fig.colorbar(im_C, ax=ax_C)

    # phi
    ax_phi = fig.add_subplot(gs[1, 2])
    im_phi = ax_phi.imshow(phi, extent=[x_min, x_max, y_min, y_max],
                           origin='lower', cmap='plasma', interpolation='bicubic', aspect='auto')
    ax_phi.set_title(r'Neutron flux $\phi$')
    ax_phi.set_xlabel('x [m]')
    ax_phi.set_ylabel('y [m]')
    fig.colorbar(im_phi, ax=ax_phi)

    # S_ext
    ax_S = fig.add_subplot(gs[2, 0])
    im_S = ax_S.imshow(S_ext, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='inferno', interpolation='bicubic', aspect='auto')
    ax_S.set_title(r'External source $S_{\mathrm{ext}}$')
    ax_S.set_xlabel('x [m]')
    ax_S.set_ylabel('y [m]')
    fig.colorbar(im_S, ax=ax_S)

    # phi 中心线 / 峰值位置信息
    ax_line = fig.add_subplot(gs[2, 1:])
    mid_y = ny // 2
    ax_line.plot(X[mid_y, :], phi[mid_y, :], 'b-', lw=2, label=f'phi at y={y[mid_y]:.2f}')
    ax_line.set_xlabel('x [m]')
    ax_line.set_ylabel(r'$\phi$')
    ax_line.set_title('Center-line neutron flux')
    ax_line.legend()
    ax_line.grid(True, ls='--', alpha=0.5)

    plt.savefig('solve_2d_coupled_phi_T_C.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nFinal coupled fields:")
    print(f"T  min={T.min():.2f}, max={T.max():.2f}, mean={T.mean():.2f}")
    print(f"C  min={C.min():.4e}, max={C.max():.4e}, mean={C.mean():.4e}")
    print(f"phi min={phi.min():.4e}, max={phi.max():.4e}, mean={phi.mean():.4e}")

if __name__ == "__main__":
    # test_solve_2d_coupled_phi_T_C()
    # test_solve_2d_T_C_2_phi(nx=201, ny=101)
    # generate_phi_10000_samples()
    # generate_T_10000_samples()
    # generate_C_10000_samples()
    T_true, C_true, phi_true = generate_true_solutions()
