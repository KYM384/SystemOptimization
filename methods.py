import numpy as np
import functools

import utils


def SGD(f, x_init, threshold=1e-2, use_armijo=False):
    """
    再急降下法
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        threshold (float): 終了条件における勾配の大きさの最大値
        use_armijo (bool): アルミホ条件を使用する(Falseならウルフ条件)
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    x = x_init
    history = [x]

    grad_f = utils.grad(f, x)
    while np.sqrt(np.sum(grad_f ** 2)) > threshold:
        d = -grad_f

        if use_armijo:
            alpha = utils.condition_armijo(f, x, d)
        else:
            alpha = utils.condition_wolfe(f, x, d)

        x = x + alpha*d
        grad_f = utils.grad(f, x)
        history.append(x)

    return x, np.array(history)


def Newton(f, x_init, threshold=1e-2, use_armijo=False):
    """
    ニュートン法
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        threshold (float): 終了条件における勾配の大きさの最大値
        use_armijo (bool): アルミホ条件を使用する(Falseならウルフ条件)
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    x = x_init
    history = [x]

    grad_f = utils.grad(f, x)
    while np.sqrt(np.sum(grad_f ** 2)) > threshold:
        d = - np.linalg.inv(utils.hessian(f, x)) @ grad_f

        if use_armijo:
            alpha = utils.condition_armijo(f, x, d)
        else:
            alpha = utils.condition_wolfe(f, x, d)

        x = x + alpha*d
        grad_f = utils.grad(f, x)
        history.append(x)

    return x, np.array(history)


def quasi_Newton(f, x_init, B_init=None, threshold=1e-2, use_armijo=False):
    """
    準ニュートン法
    Args:
        f (function): nx1の縦ベクトルを引数に取る関数
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        B_init (np.ndarray): Bの初期値; nxnの行列
        threshold (float): 終了条件における勾配の大きさの最大値
        use_armijo (bool): アルミホ条件を使用する(Falseならウルフ条件)
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    x = x_init
    history = [x]

    if B_init is None:
        B = np.identity(len(x))
    else:
        B = B_init

    s = None
    y = None

    grad_f = utils.grad(f, x)
    while np.sqrt(np.sum(grad_f ** 2)) > threshold:
        d = - np.linalg.inv(B) @ grad_f

        if use_armijo:
            alpha = utils.condition_armijo(f, x, d)
        else:
            alpha = utils.condition_wolfe(f, x, d)

        x = x + alpha*d

        next_grad_f = utils.grad(f, x)
        s = alpha * d
        y = next_grad_f - grad_f
        Bs = B @ s
        B = B - Bs@Bs.T / (s.T@Bs).item() + y@y.T / (s.T@y).item()
        grad_f = next_grad_f
        history.append(x)

    return x, np.array(history)


def active_set(Q, c, A, b, x_init):
    """
    凸2次計画法に対する有効制約法
    min \frac{1}{2}x^TQx + c^Tx
    Subject to a^T_i x > b_i  for each i=1...m
    Args:
        Q (np.ndarray): 上式のQ; nxn行列
        c (np.ndarray): 上式のc; nx1行列
        A (np.ndarray): 上式のa; mxn行列
        b (np.ndarray): 上式のb; mx1行列
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
    Returns:
        x (np.ndarray): f(x)が最小となるx
        history (np.ndarray): xの更新履歴
    """

    assert np.all(Q >= 0), "Qが半正定値ではありません"
    assert np.all(A @ x_init >= b), "x_initが実行可能解でありません"

    x = x_init
    history = [x]
    m, n = A.shape

    is_active = [utils.equal(A[i]@x, b[i]) for i in range(m)]

    while True:
        # 最適性の十分条件を満たす\bar{x}, \bar{u}を求める
        # Qx + c - \sum_{i \in I(x)} u_i a_i = 0
        # a^T_i x = b_i  for i in I(x)
        solve_A = np.zeros((n+sum(is_active), n+sum(is_active)))
        solve_b = np.zeros((n+sum(is_active), 1))

        solve_A[:n, :n] = Q
        solve_b[:n] = -c[:]
        k = 0
        for i in range(m):
            if not is_active[i]:
                continue
            solve_A[:n, n+k] = -A[i]
            solve_A[n+k, :n] = A[i]
            solve_b[n+k] = b[i]
            k += 1

        x_bar, u_bar = np.split(np.linalg.inv(solve_A) @ solve_b, [n])

        if utils.equal(x, x_bar):
            if np.all(u_bar >= 0):
                break
            else:
                k = 0
                for i in range(m):
                    if is_active[i]:
                        if u_bar[k] < 0:
                            is_active[i] = False
                            break
                        k += 1
        else:
            if np.all(A @ x_bar >= b):
                x = x_bar
            else:
                for alpha in np.linspace(0, 1, 100)[::-1]:
                    x_inter = alpha * x_bar + (1-alpha) * x
                    if np.all(A @ x_inter >= b):
                        x = x_inter
                        break

            is_active = [utils.equal(A[i]@x, b[i]) for i in range(m)]

        history.append(x)

    return x, np.array(history)


def penalty_function(f, equality, inequality, x_init, rho_init=1, threshold=1e-6, beta=2):
    """
    ペナルティ関数法
    min f(x)
    Subject to g_i(x) <= 0  for each i=1...m
    Args:
        f (function): 最小化したい関数
        equality(List[function]): 等式制約
        inequality(List[function]): 不等式制約
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        rho_init (float): \rhoの初期値
        threshold (float): 終了条件における \rho g(x) の閾値
        beta (float): \rho の更新係数
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    gs = [functools.partial(lambda g,x: g(x)**2, g) for g in equality]
    gs += [functools.partial(lambda g,x: max(g(x), 0)**2, g) for g in inequality]

    x = x_init
    rho = rho_init
    history = [x]
    
    while True:
        f_rho = lambda x: f(x) + rho * sum([g(x) for g in gs])
        x, _ = quasi_Newton(f_rho, x)
        history.append(x)

        if rho * sum([g(x) for g in gs]) < threshold:
            break

        rho *= beta

    return x, np.array(history)


def barrier_function(f, equality, inequality, x_init, rho_init=1, threshold=1e-6, beta=0.9):
    """
    バリア関数法
    min f(x)
    Subject to g_i(x) <= 0  for each i=1...m
    Args:
        f (function): 最小化したい関数
        equality(List[function]): 等式制約
        inequality(List[function]): 不等式制約
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        rho_init (float): \rhoの初期値
        threshold (float): 終了条件における \rho g(x) の閾値
        beta (float): \rho の更新係数
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    gs = [functools.partial(lambda g,x: g(x)**100, g) for g in equality]
    gs += [functools.partial(lambda g,x: -1/g(x), g) for g in inequality]

    x = x_init
    rho = rho_init
    history = [x]
    
    while True:
        f_rho = lambda x: f(x) + rho * sum([g(x) for g in gs])
        x, _ = quasi_Newton(f_rho, x)
        history.append(x)

        if rho * sum([g(x) for g in gs]) < threshold:
            break

        rho *= beta

    return x, np.array(history)


def augmented_Lagrangian(f, equality, x_init, u_init=None, rho_init=1, threshold=1e-6, beta=2):
    """
    拡張ラグランジュ法 (乗数法)
    min f(x)
    Subject to g_i(x) = 0  for each i=1...m
    Args:
        f (function): 最小化したい関数
        equality(List[function]): 等式制約
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        u_init (np.ndarray): ラグランジュ乗数uの初期値; nx1の縦ベクトル
        rho_init (float): \rhoの初期値
        threshold (float): 終了条件における \rho g(x) の閾値
        beta (float): \rho の更新係数
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    x = x_init
    if u_init is None:
        u = np.zeros((len(equality), 1))
    else:
        u = u_init
    rho = rho_init
    history = [x]

    while True:
        L_rho = lambda x: f(x) + sum([u_i*g(x) + rho/2*g(x)**2 for g, u_i in zip(equality, u)])
        x, _ = quasi_Newton(L_rho, x)
        history.append(x)

        if rho * sum([g(x) for g in equality]) < threshold:
            break

        u = u + rho * np.array([g(x) for g in equality]).reshape(-1, 1)
        rho *= beta

    return x, np.array(history)


def interior_point_point(f, equality, inequality, x_init, s_init=None, u_init=None, rho_init=1, eta=0.1, delta=0.1, threshold=1e-6, beta=0.9, use_armijo=False):
    """
    内点法
    min f(x)
    Subject to g_i(x) <= 0  for each i=1...m
    Args:
        f (function): 最小化したい関数
        equality(List[function]): 等式制約
        inequality(List[function]): 不等式制約
        x_init (np.ndarray): xの初期値; nx1の縦ベクトル
        s_init (np.ndarray): スラック変数sの初期値; nx1の縦ベクトル
        u_init (np.ndarray): ラグランジュ乗数uの初期値; nx1の縦ベクトル
        rho_init (float): \rhoの初期値
        eta (float): メリット関数の重み係数
        delta (float): \rhoの更新係数
        threshold (float): 終了条件における \rho g(x) の閾値
        beta (float): \rho の更新係数
        use_armijo (bool): アルミホ条件を使用する(Falseならウルフ条件)
    Returns:
        x (np.ndarray): f(x)が局所最小となるx
        history (np.ndarray): xの更新履歴
    """
    gs = inequality + equality

    x = x_init
    if s_init is None:
        s = np.array([-g(x) for g in inequality]).reshape(-1, 1)
    else:
        s = s_init
    if u_init is None:
        u = np.zeros((len(gs), 1))
    else:
        u = u_init
    rho = rho_init
    n = len(x)
    l = len(inequality)
    m = len(gs)
    history = [x]

    def merit_func(xs, rho):
        x, s = np.split(xs, [n])
        v = f(x)
        for i in range(m):
            if i < l:
                v -= rho * np.log(s[i]).item()
                v += eta * np.abs(gs[i](x)+s[i]).item()
            else:
                v += eta * np.abs(gs[i](x)).item()
        return v


    while rho > threshold:
        solve_A = np.zeros((n+l+m, n+l+m))
        solve_b = np.zeros((n+l+m, 1))

        grad_f = utils.grad(f, x)
        hessian_f = utils.hessian(f, x)
        grad_gs = [utils.grad(g, x) for g in gs]
        hessian_gs = [utils.hessian(g, x) for g in gs]

        solve_A[:n, :n] = hessian_f
        solve_b[:n] = -grad_f
        for i in range(m):
            solve_A[:n, :n] += u[i,0] * hessian_gs[i]
            solve_A[:n, n+l+i] = grad_gs[i][:,0]
            solve_b[:n] -= u[i] * grad_gs[i]

            if i < l:
                solve_A[n+i, n+i] = u[i,0]
                solve_A[n+i, n+l+i] = s[i,0]
                solve_b[n+i] = rho - u[i]*s[i]

                solve_A[n+l+i, :n] = grad_gs[i][:,0]
                solve_A[n+l+i, n+i] = 1
                solve_b[n+l+i] = -gs[i](x) - s[i]
            else:
                solve_A[n+l+i, :n] = grad_gs[i][:,0]
                solve_b[n+l+i] = -gs[i](x)

        dx, ds, du = np.split(np.linalg.inv(solve_A) @ solve_b, [n, n+l])

        alpha = 1
        beta = 0.9
        while np.any(u + alpha*du <= 0) or np.any(s + alpha*ds <= 0):
            alpha *= beta

        phi_eta = functools.partial(merit_func, rho=rho)
        if use_armijo:
            alpha = utils.condition_armijo(phi_eta, np.vstack([x,s]), np.vstack([dx,ds]), alpha_init=alpha)
        else:
            alpha = utils.condition_wolfe(phi_eta, np.vstack([x,s]), np.vstack([dx,ds]), alpha_init=alpha)

        x = x + alpha * dx
        s = s + alpha * ds
        u = u + alpha * du
        rho = delta * (u[:l].T @ s).item() / l

        history.append(x)

    return x, np.array(history)
