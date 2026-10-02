# 线性回归 optimized：六组模型

独立实现上层 `../main.py` 的 OLS、Ridge、Lasso 和三种 SGD 回归。使用 `load_diabetes(scaled=False)`，按 `test_size=0.2`、`random_state=42` 划分，仅用训练集拟合 `StandardScaler`；训练集为 353×10，测试集为 89×10。

## 运行与支持范围

在项目根目录运行：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python LR/optimized/main.py
```

| 文件 | 内容 |
|---|---|
| `linear.py` | OLS、Ridge、Lasso |
| `sgd.py` | 一个 SGD 类，支持 `penalty=None/"l2"/"l1"` |
| `main.py` | 独立准备数据、构造模型并对照 sklearn |

模型提供 `fit(X, y)`、`predict(X)`，保存 `coef_`、标量 `intercept_`，迭代模型另有 `n_iter_`，Lasso 另有 `dual_gap_`。仅支持本例的稠密浮点数组、单目标、拟合截距和一次性训练；默认输入、参数、调用顺序有效，不做校验或异常处理。训练预测手动使用 Python/NumPy/SciPy，不依赖其它实现目录；sklearn 只用于数据、预处理和对照。

## 相对 basic 的变化

三种常规求解均先中心化，训练后用 `y_mean - X_mean @ coef` 恢复截距，截距不参与正则化。

| 模型 | basic | 本目录参考 sklearn 的变化 |
|---|---|---|
| OLS | 构造正规方程后 `np.linalg.solve` | 直接 `scipy.linalg.lstsq(Xc, yc, cond=1e-6)`，不构造 OLS 的 Gram 矩阵；`cond` 对应 sklearn 1.9.1 默认 `tol` |
| Ridge | 通用线性系统求解 | 构造 `Xc.T @ Xc`，给对角线加 `alpha`，用 `scipy.linalg.solve(..., assume_a="pos")` 利用正定结构 |
| Lasso | 每坐标重算 `yc - Xc @ w` 和列范数，只检查绝对系数变化 | 缓存列平方范数；用 `residual -= (new - old) * column` 维护残差；相对系数变化足够小时再检查对偶间隙 |
| SGD-L2 | 每步显式加 `alpha * w` 到梯度 | 令实际系数 `w = scale * v`，收缩时只更新 `scale`，梯度更新加到 `v`；`scale < 1e-9` 时物化到 `v` 并重置为 1 |
| SGD-L1 | 使用 `alpha * sign(w)` 普通次梯度 | 使用 sklearn 的累计截断：`u` 累计应收缩量，`q` 记录各坐标实际收缩；按截断前的符号选择截断分支，可得到精确零系数 |

SGD-none 保留以完整对照上层六组，本身没有新增算法优化。这里不声称速度提升或数值稳定性改善，验证侧重计算结果和更新等价性。

**Lasso：** 内部使用未除以样本数的目标 `0.5 * (residual @ residual) + m * alpha * abs(w).sum()`。每轮检查最大系数变化是否不超过 `tol * max(abs(w))`；满足时（或最后一轮）构造可行对偶点，检查 `gap <= tol * (yc @ yc)`。对偶点是 `scale * residual`，其中 `scale = min(1, m * alpha / max(abs(Xc.T @ residual)))`；保存的 `dual_gap_ = gap / m` 与 sklearn 的尺度一致。

**SGD：** 零系数、零截距初始化，每轮用 `np.random.default_rng(42)` 打乱。累计步数从 1 开始，学习率为 `0.01 / t**0.25`，固定训练 10000 轮，不提前停止。平方误差梯度使用更新前的预测，随后处理 L2 收缩、损失更新和 L1 截断；截距仅做损失更新。L2 收缩因子为 `max(0, 1 - eta * alpha)`；L1 的 `scale` 始终为 1，稠密路径每次截断全部坐标。

**目标及参数：** 统一对照目标为 `SSE/(2m) + 正则项`。L2 正则项为 `0.1 * (w @ w) / 2`，因此 Ridge 的 `alpha=m*0.1=35.3`，SGD-L2 的 `alpha=0.1`。L1 正则项为 `1.0 * abs(w).sum()`，Lasso 与 SGD-L1 的 `alpha=1.0`；Lasso 使用 `max_iter=10000`、`tol=1e-8`。

## 实际对照与差异

`main.py` 打印双方测试 MSE/R²、非零系数数、迭代次数、最大测试预测差和统一训练目标。在 `DLdemo`、sklearn 1.9.1 下运行结果：

| 模型 | optimized MSE | sklearn MSE | 最大测试预测差 | 非零数（O/SK） | 迭代数（O/SK） |
|---|---:|---:|---:|---:|---:|
| OLS | 2900.193628 | 2900.193628 | 0 | 10/10 | — |
| Ridge-L2 | 2859.981265 | 2859.981265 | 0 | 10/10 | — |
| Lasso-L1 | 2824.622659 | 2824.622659 | 5.68e-14 | 9/9 | 130/130 |
| SGD-none | 2900.491042 | 2901.199288 | 0.198820 | 10/10 | 10000/10000 |
| SGD-L2 | 2859.818165 | 2861.072009 | 0.252814 | 10/10 | 10000/10000 |
| SGD-L1 | 2824.755719 | 2825.660006 | 0.205432 | 9/9 | 10000/10000 |

OLS/Ridge 的系数、截距、预测通过 `atol=rtol=1e-8` 核对；Lasso 通过 `atol=1e-5`、`rtol=1e-6` 核对。Lasso 的 `dual_gap_` 为 `1.52614e-5`，小于本例阈值 `6.07640e-5`，并用重新计算的残差独立核对了间隙。小数据短程验证中，L2 延迟缩放与直接逐步更新的最大系数差不超过 `1.67e-16`，覆盖小缩放物化和收缩因子为零；L1 与逐坐标累计截断参考结果一致。

- **Lasso：** sklearn 1.9.1 已采用 Gap Safe Screening，本目录为保持简洁没有实现该特征筛选，也不支持随机坐标顺序、预计算 Gram 矩阵、正系数约束等。本例迭代数相同，其它数据上可能不同。
- **SGD：** 相同种子不代表相同样本顺序，NumPy 的打乱与 sklearn 内部打乱不同，因此不要求最终参数逐位相同。只支持上述平方误差、固定轮数、每轮打乱和 `invscaling`；不实现其它损失/学习率、稀疏输入、样本权重、提前停止、平均系数、增量训练或梯度裁剪。

## 源码依据

参考 **scikit-learn 1.9.1**，本地包目录为 `/home/fire/miniconda3/envs/DLdemo/lib/python3.13/site-packages/sklearn`。以下位置均相对此目录；只参考算法与数学流程，实际代码为 Python。

| 内容 | 源码位置 |
|---|---|
| 中心化、截距恢复、稠密最小二乘 | `linear_model/_base.py`：`_preprocess_data`、`LinearRegression.fit`、`_set_intercept` |
| Ridge 正定求解 | `linear_model/_ridge.py`：`_solve_cholesky` |
| Lasso 参数尺度、残差维护、列范数与间隙 | `linear_model/_coordinate_descent.py`、`linear_model/_cd_fast.pyx`：`enet_coordinate_descent`、`gap_enet`、`dual_gap_formulation_A` |
| SGD 学习率、更新顺序、累计截断 | `linear_model/_stochastic_gradient.py`、`linear_model/_sgd_fast.pyx.tp`：`_plain_sgd`、`l1penalty` |
| L2 延迟缩放及物化阈值 | `utils/_weight_vector.pyx.tp`：`add`、`dot`、`scale`、`reset_wscale`（float64 路径） |
