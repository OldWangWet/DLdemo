# 线性回归 sklearn_impl

参考 **scikit-learn 1.9.1** 的 CPU NumPy/SciPy 路径，独立实现 `LinearRegression`、`Ridge`、`Lasso`、`SGDRegressor`，覆盖上层 `../main.py` 的六组模型。训练与预测不调用 sklearn 目标模型或其内部优化内核；sklearn 模型仅在 `main.py` 中用于对照。

## 运行与文件

在项目根目录运行六组 diabetes 示例：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python LR/sklearn_impl/main.py
```

只运行小数据能力核对：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python LR/sklearn_impl/main.py --check-capabilities
```

也可从项目根目录导入：`from LR.sklearn_impl import LinearRegression, Ridge, Lasso, SGDRegressor`。

| 文件 | 内容 |
|---|---|
| `common.py` | 公共校验、加权中心化、权重缩放、稀疏中心化算子、截距与预测 |
| `linear.py` | 稠密最小二乘、稀疏 LSQR、稠密非负最小二乘 |
| `ridge.py` | Cholesky、SVD、LSQR、共轭梯度与自动选择 |
| `lasso.py` | 手写稠密、CSC、Gram 坐标下降及对偶间隙 |
| `sgd.py` | 损失、学习率、惰性 L2、累计 L1、平均与增量状态 |
| `random_utils.py` | 手写 XorShift 和 Fisher–Yates；文件名避免遮蔽标准库 `_random` |
| `__init__.py` | 导出四个模型 |
| `main.py` | 六组原例、sklearn 参考模型与有效输入能力核对 |

本目录不导入或依赖 `basic/`、`optimized/`。复用 sklearn 公共基类与 `validate_data`、`check_is_fitted`、`check_random_state` 等工具；预测的矩阵乘法与截距相加自行实现。

## 实际支持范围

所有模型提供 `fit`、`predict`；公共基类同时提供 `get_params`、`set_params`、`score`。输入使用 float64 计算，整数和 float32 输入会转换，不复现 float32 专用数值路径。支持 NumPy 稠密数组及下表列出的 SciPy 稀疏表示。

| 模型 | 主要参数 | 已实现能力 |
|---|---|---|
| `LinearRegression` | `fit_intercept`、`copy_X`、`tol`、`positive` | 样本权重、稠密/稀疏、单/多目标；稠密正系数；稠密非正约束路径的 `rank_`、`singular_` |
| `Ridge` | `alpha`、`fit_intercept`、`copy_X`、`max_iter`、`tol`、`solver`、`positive=False` | `auto/cholesky/svd/lsqr/sparse_cg`；权重、单/多目标；每目标不同 `alpha`；`solver_`、LSQR 的 `n_iter_` |
| `Lasso` | `alpha`、`fit_intercept`、`precompute`、`copy_X`、`max_iter`、`tol`、`warm_start`、`positive`、`random_state`、`selection` | 权重、稠密/CSC、多目标分别求解；循环/随机坐标；布尔或数组 Gram；`dual_gap_`、`n_iter_`、`sparse_coef_` |
| `SGDRegressor` | `loss`、`penalty`、`alpha`、`l1_ratio`、`fit_intercept`、`max_iter`、`tol`、`shuffle`、`epsilon`、`random_state`、`learning_rate`、`eta0`、`power_t`、`n_iter_no_change`、`warm_start`、`average` | 稠密/CSR、样本权重、四种损失、四种正则、四种常规学习率；`coef_init`、`intercept_init`；`partial_fit`、平均参数、`t_` |

- **稀疏与正系数：** OLS 的正系数路径只支持稠密输入。Ridge 的 SVD 不支持稀疏输入；稀疏加截距使用 `auto/lsqr/sparse_cg`，稀疏 Cholesky 要求 `fit_intercept=False`。CSC/CSR 在训练时使用对应表示，稀疏中心化不生成完整稠密特征矩阵；Cholesky 与 Gram 路径会生成稠密乘积矩阵。
- **权重：** OLS/Ridge 不归一化权重，中心化后用平方根权重缩放；Lasso 先归一化至权重总和等于样本数；SGD 直接把每个样本的数据更新乘权重。权重要求有限、非负且总和大于零。
- **目标：** OLS/Ridge/Lasso 支持多目标；Lasso 是各目标独立的 L1 回归，不是 MultiTask Lasso。SGD 只支持单目标，与参考模型一致。
- **SGD 选项：** 损失为 `squared_error/huber/epsilon_insensitive/squared_epsilon_insensitive`；正则为 `None/l1/l2/elasticnet`；学习率为 `constant/invscaling/optimal/adaptive`。`average=True` 从第一步平均，正整数指定平均起点。

## 与源码对应的计算

**OLS：** 加权中心化后，稠密数据用 SciPy `lstsq(cond=tol)`；稀疏数据用手写中心化 `LinearOperator` 加 SciPy `lsqr(atol=tol, btol=tol)`；正系数使用 SciPy `nnls`。`tol` 影响稠密路径是 1.9 的变化。

**Ridge：** 目标为 `SSE + alpha * ||w||²`。稠密 `auto` 选择 Cholesky，稀疏选择共轭梯度。Cholesky 在特征多于样本时求解样本空间方程；多目标相同惩罚共享一次矩阵求解。SVD 使用 `s/(s²+alpha)` 及源码的 `s>1e-15` 掩码；LSQR 使用 `damp=sqrt(alpha)`；CG 用算子表示正规方程。没有显式矩阵求逆。

**Lasso：** 目标为 `SSE/(2m) + alpha * ||w||₁`。维护残差并预计算列平方范数，内部软阈值为 `m*alpha`。CSC 的加权残差和均值补偿处理隐式零元素；Gram 路径维护相关量。相对系数变化触发对偶间隙检查，保存除以样本数后的 `dual_gap_`；零初始化已满足间隙时可使用零轮。`alpha=0` 按源码用一阶条件代替 L1 对偶间隙。

**SGD：** 逐样本更新；L2 先进行惰性缩放，L1/Elastic Net 使用 `u/q` 累计截断，不能用普通 `sign(w)` 次梯度替代。每轮以相同局部 seed 继续打乱已有索引，使用源码 XorShift/Fisher–Yates。稀疏截距更新乘 `0.01`。损失导数限制在 ±`1e12`；平均保存标准系数、平均缓冲与惰性平均状态，按源码在 L1 截断之前更新。`fit` 重置步数，warm start 保留训练参数；`partial_fit` 只运行一轮并延续步数，L1 累计辅助量每次训练调用重新初始化。

SciPy 通用线性代数/NNLS 求解器的使用与参考源码对应；坐标下降、随机更新、损失与平均状态全部用 Python/NumPy 写出。

## 差异与未实现内容

- 未实现 OLS 的 `n_jobs` 并行调度；多目标串行处理。
- 未实现 Ridge `sag/saga/lbfgs` 和 `positive=True`，不支持的 solver 明确报错。`max_iter`/`tol` 对直接求解不生效；CG 样本空间分支与 1.9.1 源码一样不传 `max_iter`。
- Lasso 未实现 1.9.1 默认启用的 Gap Safe Screening；收敛解可对齐，但迭代轮数和随机坐标轨迹不保证相同。稀疏输入不使用 Gram；稠密传入 Gram 后若中心化或加权改变数据则重新计算。
- SGD 不包含 `pa1/pa2`、验证集停止 `early_stopping=True`、`validation_fraction` 或详细 `verbose` 输出。`tol=None` 固定训练轮数；有限 `tol` 使用 1.9.1 的逐步目标平均值，包含正则项，adaptive 停滞时步长除以五。
- SGD 停止条件保留 1.9.1 的实际范数缓存规则：`WeightVector.add` 只累计当前存储坐标并覆盖缓存，L1 截断不刷新缓存，目标累计不乘样本权重。因此该停止累计值不能解释为每步完整权重向量的精确加权目标。
- warm start 已有 SGD 模型时，沿用源码 `_partial_fit` 的行为，显式初始化参数只在首次分配系数时使用。未覆盖训练后动态切换 `average` 等状态参数的组合。
- 不提供正则路径/CV、完整 estimator tags、元数据路由、Array API/GPU、回调、HTML、序列化兼容或逐字异常/警告兼容；没有性能或内存改善声明。

## 验证结果

在 `DLdemo` 的 sklearn 1.9.1 下，原始 diabetes 特征按 `test_size=0.2`、`random_state=42` 划分，仅用训练集拟合 `StandardScaler`。353 个训练样本、10 个特征、89 个测试样本；六组参数与上层 `main.py` 相同。

| 模型 | 手写测试 MSE | sklearn 测试 MSE | 最大测试预测差 | 手写/参考轮数 | 非零系数 |
|---|---:|---:|---:|---|---:|
| OLS | 2900.193628 | 2900.193628 | 0 | — | 10 |
| Ridge-L2 | 2859.981265 | 2859.981265 | 0 | — | 10 |
| Lasso-L1 | 2824.622659 | 2824.622659 | 5.68e-14 | 130 / 130 | 9 |
| SGD-none | 2901.199288 | 2901.199288 | 0 | 10000 / 10000 | 10 |
| SGD-L2 | 2861.072009 | 2861.072009 | 0 | 10000 / 10000 | 10 |
| SGD-L1 | 2825.660006 | 2825.660006 | 0 | 10000 / 10000 | 9 |

系数、截距、预测均通过核对：`rtol=1e-8`，OLS/Ridge `atol=1e-8`，Lasso `atol=1e-5`，SGD `atol=1e-6`。SGD 的 `n_iter_` 与 `t_` 精确一致，三组 `t_=3530001`。这些是本次环境与输入的实测结果，不保证跨版本或全部数据逐位相等。

51 组小数据能力核对全部通过，最大预测差约为 `4e-15`。检查覆盖秩亏、加权稀疏、多目标、NNLS、全部支持的 Ridge solver、宽矩阵、Lasso Gram/随机/正系数/warm start、SGD 损失/学习率/正则/平均/有限容差/增量/warm start/初始化。它只验证有效输入和已承诺路径，不是全面 sklearn 测试套件。

## 源码参考

参考包根目录：`/home/fire/miniconda3/envs/DLdemo/lib/python3.13/site-packages/sklearn`。以下均为本地 **1.9.1** 实际源码，包含 Cython 模板，仅参考计算方法，不调用其训练内核。

| 内容 | 相对源码路径与入口 |
|---|---|
| 加权中心化、截距与 OLS | `linear_model/_base.py`：`_preprocess_data`、`_rescale_data`、`LinearRegression.fit`、`_set_intercept` |
| Ridge solver、自动选择 | `linear_model/_ridge.py`：`_solve_cholesky`、`_solve_cholesky_kernel`、`_solve_svd`、`_solve_lsqr`、`_solve_sparse_cg`、`resolve_solver_for_numpy` |
| Lasso 外层、权重与预计算 | `linear_model/_coordinate_descent.py`：`ElasticNet.fit`、`Lasso`；`linear_model/_base.py`：`_pre_fit` |
| 坐标更新与间隙 | `linear_model/_cd_fast.pyx`：`enet_coordinate_descent`、`sparse_enet_coordinate_descent`、`enet_coordinate_descent_gram`、各 `gap_enet` 分支 |
| SGD 参数、fit/partial_fit、随机种子 | `linear_model/_stochastic_gradient.py`：`BaseSGDRegressor`；`linear_model/_base.py`：`make_dataset` |
| SGD 更新、学习率、L1、停止 | `linear_model/_sgd_fast.pyx.tp`：`_plain_sgd`、`l1penalty` |
| SGD 惰性系数、范数缓存、平均 | `utils/_weight_vector.pyx.tp`：`WeightVector` |
| 随机打乱 | `utils/_seq_dataset.pyx.tp`：`shuffle`；`utils/_random.pxd`：`our_rand_r` |
