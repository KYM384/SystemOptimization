import matplotlib.pyplot as plt
import numpy as np


def equal(x0, x1, delta=5e-2):
    return np.max(np.abs(x0 - x1)) < delta


def grad(f, x, delta=1e-6):
    """
    f(x)の勾配を計算する
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x (np.ndarray): nx1の縦ベクトル
        delta (float): 微小値hの大きさ
    Returns:
        grad_f (np.ndarray): nx1の縦ベクトル∇f
    """
    n = x.shape[0]
    grad_f = np.zeros((n, 1))
    for i in range(n):
        h = np.eye(1, n, i).T * delta
        grad_f[i][0] = (f(x+h) - f(x-h)) / (2*delta)
    return grad_f


# https://research.miidas.jp/2019/06/pythonでやる多次元ニュートン法/
def hessian(f, x, delta=1e-6):
    """
    f(x)のヘッセ行列を計算する
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x (np.ndarray): nx1の縦ベクトル
        delta (float): 微小値hの大きさ
    Returns:
        hessian_f (np.ndarray): nxnのヘッセ行列∇^2f
    """
    n = x.shape[0]
    hessian_f = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # f_xy(x,y) = (f_x(x,y+h) - f_x(x,y-h)) / 2h
            # f_x(x,y+h) = (f(x+h,y+h) - f(x-h,y+h)) / 2h
            # f_x(x,y-h) = (f(x+h,y-h) - f(x-h,y-h)) / 2h
            hx = np.eye(1, n, i).T * delta
            hy = np.eye(1, n, j).T * delta

            fp = (f(x+hx+hy) - f(x-hx+hy)) / (2*delta)
            fm = (f(x+hx-hy) - f(x-hx-hy)) / (2*delta)
            hessian_f[i][j] = (fp - fm) / (2*delta)
            hessian_f[j][i] = hessian_f[i][j]
    return hessian_f


def condition_armijo(f, x, d, tau=1e-4, beta=0.9, alpha_init=1.0):
    """
    g(\alpha) = f(x + \alpha d) に対して
    アルミホ条件 g(\alpha) < g(0) + \tau g'(0)\alpha を満たす\alphaを求める
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x (np.ndarray): nx1の縦ベクトル
        d (np.ndarray): 移動ベクトル; nx1の縦ベクトル
        tau (float): アルミホ条件中の\tau
        beta (float): \alpha の更新方法; \alpha_{t+1} = \beta \alpha_t と更新する
        alpha_init(float): \alpha の初期値
    Returns:
        alpha (float): アルミホ条件中の\alpha
    """
    g = lambda alpha: f(x + alpha * d)
    g0 = f(x)
    grad_g0 = grad(f, x).T @ d

    alpha = alpha_init
    # アルミホ条件を満たす最大の\alpha
    while not equal(alpha, 0) and g(alpha) > g0 + tau * grad_g0 * alpha:
        alpha *= beta

    # アルミホ条件を満たす\alphaのうち、g(\alpha)が最小になる\alpha
    new_alpha = beta * alpha
    while not equal(alpha, 0) and g(new_alpha) < g(alpha):
        alpha = new_alpha
        new_alpha *= beta

    return alpha


def condition_wolfe(f, x, d, tau1=1e-4, tau2=0.5, beta=0.9, alpha_init=1.0):
    """
    g(\alpha) = f(x + \alpha d) に対して
    ウルフ条件 g(\alpha) < g(0) + \tau_1 g'(0)\alpha かつ g'(\alpha) > \tau_2 g'(0) を満たす\alphaを求める
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x (np.ndarray): nx1の縦ベクトル
        d (np.ndarray): 移動ベクトル; nx1の縦ベクトル
        tau1 (float): ウルフ条件中の\tau_1
        tau2 (float): ウルフ条件中の\tau_2
        beta (float): \alpha の更新方法; \alpha_{t+1} = \beta \alpha_t と更新する
        alpha_init(float): \alpha の初期値
    Returns:
        alpha (float): ウルフ条件中の\alpha
    """
    g = lambda alpha: f(x + alpha * d)
    g0 = f(x)
    grad_g0 = (grad(f, x).T @ d).item()

    alpha = alpha_init
    # ウルフ条件を満たす最大の\alpha
    grad_g = lambda alpha: (grad(f, x+alpha*d).T @ d).item()

    while not equal(alpha, 0) and g(alpha) > g0 + tau1 * grad_g0 * alpha and grad_g(alpha) > tau2 * grad_g0:
        alpha *= beta

    # ウルフ条件を満たす\alphaのうち、g(\alpha)が最小になる\alpha
    new_alpha = beta * alpha
    while not equal(alpha, 0) and g(new_alpha) < g(alpha) and grad_g(new_alpha) > tau2 * grad_g0:
        alpha = new_alpha
        new_alpha *= beta

    return alpha


def visualize(f, x_history, equality=[], dim1=0, dim2=1, scale=1.5, step=50, alpha=0.3):
    """
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x_history (List[np.ndarray]): 更新されたxのリスト
        equality (List[function]): 等式制約のリスト
        dim1 (int): グラフの横軸の次元インデックス
        dim2 (int): グラフの縦軸の次元インデックス
        scale (float): 描画範囲のスケール
        step (int): 高等線の間隔
        alpha (float): 高等線の色の濃さ
    """
    n = len(x_history[-1])
    N = 100

    x0 = np.min(x_history[:, dim1])
    x1 = np.max(x_history[:, dim1])
    y0 = np.min(x_history[:, dim2])
    y1 = np.max(x_history[:, dim2])

    x_pts = np.linspace((x0+x1)/2-(x1-x0)*scale/2, (x0+x1)/2+(x1-x0)*scale/2, N)
    y_pts = np.linspace((y0+y1)/2-(y1-y0)*scale/2, (y0+y1)/2+(y1-y0)*scale/2, N)
    x_grid, y_grid = np.meshgrid(x_pts, y_pts)

    data = np.zeros((n, 1, N, N))
    data[dim1, 0] = x_grid
    data[dim2, 0] = y_grid
    for i in range(n):
        if i == dim1 or i == dim2:
            continue
        data[i] = np.full((1, N, N), x_history[-1,i])

    z = np.array([[f(data[:,:,i,j]) for j in range(N)] for i in range(N)])
    contf = plt.contourf(x_grid, y_grid, z, step, cmap="PuOr", alpha=alpha)
    plt.colorbar(contf)

    for g in equality:
        g_value = np.array([[g(data[:,:,i,j]) for j in range(N)] for i in range(N)])
        g_is0_y = y_pts[np.argmin(np.abs(g_value), axis=0)]
        plt.plot(x_pts, g_is0_y, linestyle="--")

    plt.plot(x_history[:,dim1], x_history[:,dim2], marker="o")
    plt.plot(x_history[0,dim1], x_history[0,dim2], marker="o", color="black")
    plt.plot(x_history[-1,dim1], x_history[-1,dim2], marker="o", color="red")

