# 线性回归 basic：九组模型

用于学习 `../main.py` 的 diabetes 数据：原始特征加载后，按 `test_size=0.2`、`random_state=42` 划分；只用训练集拟合 `StandardScaler`。训练集为 353 个样本、10 个特征，测试集为 89 个样本。

## 运行与结构

在项目根目录运行：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python LR/basic/main.py
```

| 文件 | 内容 |
|---|---|
| `linear.py` | OLS、Ridge、Lasso 三个类 |
| `sgd.py` | 一个 SGD 类，通过 `penalty=None/"l2"/"l1"` 构造三组模型 |
| `bgd.py` | 一个 BGD 类，同样构造三组模型 |
| `main.py` | 数据准备、九组手写模型、六组 sklearn 参考模型和对照表 |

所有模型提供 `fit(X, y)` 和 `predict(X)`；`fit` 返回自身，训练后保存 `coef_` 和标量 `intercept_`，迭代模型另有 `n_iter_`。仅支持本例的稠密数组、单目标、拟合截距和一次性训练，默认输入、参数、调用顺序有效，不做校验或异常处理。模型的训练与预测只使用 NumPy，与另外两个实现目录独立。

## 常规求解

先中心化：`Xc = X - X.mean(axis=0)`，`yc = y - y.mean()`。求解后统一恢复截距：`intercept = y.mean() - X.mean(axis=0) @ coef`，所以截距不参与正则化。

- **OLS：** 解正规方程 `Xc.T @ Xc @ w = Xc.T @ yc`，调用 `np.linalg.solve`。本例特征矩阵满秩，因此直接求解，不添加不可逆时的分支。
- **Ridge：** 解 `(Xc.T @ Xc + alpha * I) @ w = Xc.T @ yc`。目标为 `SSE + alpha * (w @ w)`。
- **Lasso：** 从零系数开始，按特征顺序循环坐标下降。固定其它系数时，先计算 `residual = yc - Xc @ w + Xc[:, j] * w[j]`，令 `rho = Xc[:, j] @ residual / m`、`z = Xc[:, j] @ Xc[:, j] / m`，再更新 `w[j] = sign(rho) * max(abs(rho) - alpha, 0) / z`。每轮最大系数变化小于 `1e-8` 时停止，上限为 10000 轮。

## BGD 与 SGD

两者都从零系数、零截距开始，使用下列统一目标：

$$
J(w,b)=\frac{1}{2m}\|Xw+b\mathbf{1}-y\|_2^2+R(w)
$$

| 正则方式 | 正则项 `R(w)` | 正则梯度 | 本例 `alpha` |
|---|---|---|---|
| 无正则 | `0` | `0` | `0.0` |
| L2 | `alpha * (w @ w) / 2` | `alpha * w` | `0.1` |
| L1 | `alpha * abs(w).sum()` | `alpha * sign(w)` | `1.0` |

**BGD：** 每次计算 `error = X @ w + b - y`，系数梯度为 `X.T @ error / m + 正则梯度`，截距梯度为 `error.mean()`。用固定学习率 `0.1` 更新 10000 次。系数和截距使用同一次更新前的残差。

本训练集的 `X.T @ X / m` 最大特征值约为 `3.96881`。无正则及 L2 的最大曲率分别约为 `3.96881`、`4.06881`，步长 `0.1` 小于对应的 `2 / 最大曲率`。最小特征值约为 `0.00923`，若无正则 BGD 使用 `0.01`，10000 次后在最慢方向仍保留约 40% 的初始参数误差；使用 `0.1` 后约为 0.01%。这是针对训练集的步长说明，不涉及测试集调参，也不用于声称 L1 收敛。

**SGD：** 每轮使用 `np.random.default_rng(42)` 打乱样本，逐样本计算 `error = x @ w + b - y`，系数梯度为 `error * x + 正则梯度`，截距梯度为 `error`。累计步数 `t` 从 1 开始，每次更新后增加，跨轮累计；学习率为 `0.01 / t**0.25`。固定遍历 10000 轮，不提前停止。

**参数换算：** Ridge 的目标除以 `2m` 后，正则项为 `alpha * (w @ w) / (2m)`，因此 Ridge 使用 `alpha=m*0.1=35.3`，对应 SGD/BGD 的 L2 `alpha=0.1`。Lasso、SGD-L1 和 BGD-L1 都使用 `SSE/(2m) + alpha * abs(w).sum()`，所以 L1 的 `alpha` 都为 `1.0`。

## 对照与差异

`main.py` 的对照表包含双方测试 MSE、R²、手写模型非零系数数及迭代次数、最大测试预测差、双方统一训练目标。BGD 分别对照 OLS、Ridge、Lasso；SGD 对照同参数的 sklearn SGD。SGD 一轮有 353 次更新，BGD 一次迭代只有一次更新，迭代数表示不同的更新过程。

- **OLS：** 本实现使用课本正规方程；sklearn 的稠密 OLS 使用 `scipy.linalg.lstsq`。本例结果应在数值误差范围内一致。
- **Lasso：** 本实现仅以最大系数变化停止；sklearn 使用相对系数变化及对偶间隙等判断。停止轮数可能不同。
- **SGD：** 固定种子让本实现可复现，但 NumPy 打乱与 sklearn 的内部打乱顺序不同，不要求最终参数完全相同。
- **L1 次梯度：** 手写 SGD/BGD 在零处选择次梯度 0，直接加上 `alpha * sign(w)`。sklearn SGD-L1 使用累计截断规则。普通次梯度通常不会给出精确零系数；固定步长的 BGD-L1 可能在零附近振荡，不保证每轮目标下降或得到精确 Lasso 解，不额外把小系数裁成零。

验证只针对这份数据：OLS/Ridge 的系数、截距、预测用 `rtol=1e-8`、`atol=1e-8` 核对；Lasso 用 `rtol=1e-6`、`atol=1e-5` 核对。梯度下降检查结果有限、训练目标低于零初始化，并记录与参考解的差异。

### 本例运行结果

在 `DLdemo` 环境、sklearn 1.9.1 下，使用上述固定数据和参数运行：

| 模型 | basic 测试 MSE | sklearn 测试 MSE | 最大测试预测差 |
|---|---:|---:|---:|
| OLS | 2900.193628 | 2900.193628 | 1.71e-13 |
| Ridge-L2 | 2859.981265 | 2859.981265 | 1.14e-13 |
| Lasso-L1 | 2824.622665 | 2824.622659 | 2.91e-6 |
| SGD-none | 2900.491042 | 2901.199288 | 0.198820 |
| SGD-L2 | 2859.818165 | 2861.072009 | 0.252814 |
| SGD-L1 | 2824.619061 | 2825.660006 | 0.241509 |
| BGD-none | 2900.190273 | 2900.193628 | 0.001164 |
| BGD-L2 | 2859.981265 | 2859.981265 | 2.56e-13 |
| BGD-L1 | 2825.238628 | 2824.622659 | 0.211917 |

OLS、Ridge、Lasso 的系数、截距、预测均通过上面的容差核对。手写 Lasso 使用 158 轮，sklearn 使用 130 轮；两者都有 9 个非零系数。手写 SGD-L1 和 BGD-L1 都有 10 个非零系数，反映普通次梯度与软阈值/累计截断的区别。

六组梯度下降模型均完成 10000 次对应的迭代，结果有限，统一训练目标均小于零初始化的 `14855.661473`。其中 BGD-L1 的最终目标为 `1547.939133`，Lasso 参考目标为 `1547.877345`，属于固定步长次梯度的近似结果。三种 SGD 另外用 20 轮检查了固定种子的可复现性。

## 源码参考

参考本项目 Conda 环境中的 **scikit-learn 1.9.1**。以下位置相对于 sklearn 包目录；参考算法和数学流程，实际模型代码用 Python/NumPy 实现。

| 内容 | 源码位置 |
|---|---|
| 中心化、截距恢复、OLS | `linear_model/_base.py`：`_preprocess_data`、`LinearRegression.fit`、`_set_intercept` |
| Ridge 正规方程 | `linear_model/_ridge.py`：`_solve_cholesky` |
| Lasso 坐标更新及软阈值 | `linear_model/_coordinate_descent.py`、`linear_model/_cd_fast.pyx`：`enet_coordinate_descent` |
| SGD 训练参数、逐样本更新、学习率、L1 截断 | `linear_model/_stochastic_gradient.py`、`linear_model/_sgd_fast.pyx.tp`：`_fit_regressor`、`_plain_sgd`、`l1penalty` |

BGD 按上层 [算法说明](../readme.md) 的整批梯度公式实现；它是本次额外要求的课本流程，不对应 sklearn 的单独 BGD 模型。

- [sklearn 线性模型说明](https://scikit-learn.org/stable/modules/linear_model.html)
- [sklearn SGD 数学说明](https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation)
