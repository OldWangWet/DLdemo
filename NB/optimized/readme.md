# 优化版朴素贝叶斯：算法与数学计算

本目录对应[项目 README](../../readme.md) 的第二阶段：以 [basic](../basic/nb.py) 为基线，参考 sklearn 已实现的算法与数学方法调整计算流程，说明依据、收益和适用条件，并验证结果。模型代码在 [nb.py](nb.py)，模型计算使用 NumPy/SciPy，sklearn 仅用于数据加载和对照。

参考版本为 `DLdemo` 环境中的 scikit-learn 1.9.1，源码为 `/home/fire/miniconda3/envs/DLdemo/lib/python3.13/site-packages/sklearn/naive_bayes.py`。下文标注该文件中的类和方法；原理及手算示例见 [NB README](../readme.md)。本目录实现源码计算方法的稠密、无样本权重版本，保留精简接口，并不复现全部 sklearn 行为。

## 1. 阶段边界与实际增量

本阶段支持稠密输入和一次性 `fit(X, y)`。主要内容为批量计数与评分、对数之差、伯努利代数变形、稳定概率归一化，以及高斯统计量合并公式。等价变形应保留分类规则，浮点运算顺序改变允许合理误差；优化不保证提高准确率或所有规模的速度。

CSR 路径、增量训练接口、样本权重、完整 estimator 协议、参数管理、输入输出及异常兼容，均留给第三阶段 `sklearn_impl/`。本目录没有实现该阶段，也不把 CPU 线程设置、稀疏存储或输入规范化列为算法数学优化。

| 内容 | basic 现状 | optimized 的实际增量 |
|---|---|---|
| 对数域评分与直接取最大值 | 已实现 | 沿用，不计为新增优化 |
| 离散条件对数概率表 | 已保存 | 多项式、伯努利、类别改用对数之差；矩阵评分或批量查表 |
| 高斯方差与离散计数平滑 | 已实现 | 沿用，并补充源码可选的小平滑值数值保护 |
| 类别直方图 | 已用 `np.bincount` | 保留训练方式，预测改为按特征查出全部类别 |
| 高斯逐类预测 | 已实现 | 固定对数项与样本偏差项分别求和；保留均值方差合并公式 |
| 补集计数 | 已用总计数减本类计数 | 分离计数和参数更新，避免生成随后被覆盖的多项式概率；矩阵评分 |
| 稳定概率归一化 | 未提供概率输出 | 使用 `logsumexp`；输出方法用于调用和验证该计算 |

公共基类、类别映射、基本输入检查、返回数组及文件组织是实现所需的接口安排，不作为优化收益。

### 方法与源码对应

| 计算方法 | sklearn 1.9.1 对应方法 | 本阶段归属 |
|---|---|---|
| 指示矩阵计数、计数与参数更新分离 | `_BaseDiscreteNB.fit`；三种模型的 `_count` | 批量计算流程 |
| 平滑概率使用对数之差 | `MultinomialNB`、`BernoulliNB`、`CategoricalNB` 的 `_update_feature_log_prob` | 数值计算 |
| 矩阵评分 | `MultinomialNB`、`ComplementNB` 的 `_joint_log_likelihood` | 批量计算流程 |
| 伯努利未出现项变形 | `BernoulliNB._joint_log_likelihood` | 数学等价变形 |
| 类别批量查表 | `CategoricalNB._joint_log_likelihood` | 批量计算流程 |
| 高斯统计量合并 | `GaussianNB._update_mean_variance` | 数学方法；没有增量训练接口 |
| 高斯评分分项求和 | `GaussianNB._joint_log_likelihood` | 等价计算结构 |
| 直接生成补集权重 | `ComplementNB._update_feature_log_prob` | 避免无用参数计算 |
| 小平滑值可选下限 | `_BaseDiscreteNB._check_alpha` | 数值保护；会改变有效平滑值 |
| `logsumexp` 归一化 | `_BaseNB.predict_log_proba` | 数值稳定性 |
| 补集 `norm=True` | `ComplementNB._update_feature_log_prob` | 数学算法选项，会改变分类规则 |

补集归一化与平滑值截断单独对照相同参数的 sklearn，不能把它们当作保持结果不变的加速。

## 2. 文件与最小调用接口

```text
NB/optimized/
├── __init__.py              # 导出五种模型
├── nb.py                    # 模型计算与基本检查
├── main.py                  # digits 上 basic、optimized、sklearn 三方对照
├── test_nb.py               # 公式、参数与数值边界验证
├── benchmark.py             # 相同稠密输入的 fit/predict 计时
├── benchmark_results.json   # 当前代码的 15 组实测记录
└── readme.md
```

提供 `fit`、`predict`、`joint_log_likelihood`、`predict_log_proba` 和 `predict_proba`。后面三个方法用于检查评分和稳定归一化；本目录的 `joint_log_likelihood` 对应 sklearn 的公共方法 `predict_joint_log_proba`，不声称实现了其完整接口。

```python
from NB.optimized import MultinomialNB

model = MultinomialNB(alpha=1.0).fit([[3, 1], [0, 4]], ["垃圾", "正常"])
labels = model.predict([[2, 0]])
scores = model.joint_log_likelihood([[2, 0]])
proba = model.predict_proba([[2, 0]])
```

类别由 `np.unique` 排序，参数表和得分列遵循 `classes_` 顺序。标签可以为不连续整数或字符串。重复调用 `fit` 重新初始化统计量，不累加历史数据。

基本边界与基础版一致：有限二维稠密特征、非空训练集、一维标签、特征数一致；类别编码为非负整数，伯努利 `binarize=None` 时输入为 0/1，补集至少两类，高斯最终方差严格为正。标量 `alpha` 必须有限且大于零，`min_categories` 支持正整数标量。

这些检查用于使公式在有效定义域内运行，不计入算法优化。全面校验、零或向量平滑、自定义先验、完整警告异常兼容，以及退化数据上的 sklearn 无效值行为留给第三阶段。

## 3. 多项式：计数与矩阵评分

记训练样本数为 $n$、预测样本数为 $m$、特征数为 $d$、类别数为 $C$。将标签编码为指示矩阵 $Y$，每行所属类别为 1，其余为 0。计数为：

$$
N=Y^\top X,\qquad N\in\mathbb R^{C\times d}.
$$

```python
self.feature_count_ += Y.T @ X
self.class_count_ += Y.sum(axis=0)
```

对应 `MultinomialNB._count`，伯努利和补集使用相同计数式。无样本权重时，源码的 `fit` 指示矩阵为整数，本实现使用 `np.int64`。计数和参数更新分离，允许补集模型直接生成自己的权重。

**收益与条件：** 合并逐类筛选和求和的调用，但密集矩阵计数约有 $O(nCd)$ 运算和 $O(nC)$ 指示矩阵开销，基础版逐类求和的有效特征处理量约为 $O(nd)$。不能宣称矩阵计数必然更快或复杂度更低。

记有效平滑值为 $a$，对数概率按源码的对数之差计算：

$$
\log\theta_{cj}=\log(N_{cj}+a)-\log\sum_{\ell=1}^{d}(N_{c\ell}+a).
$$

它避免先求很小的概率比值再取对数造成的部分下溢，计数及求和仍需处于浮点可表示范围。

预测一次计算全部类别：

$$
L=X(\log\Theta)^\top+\log\boldsymbol\pi.
$$

对应 `MultinomialNB._joint_log_likelihood`。复杂度仍为 $O(mdC)$，计算从逐类的逐元素乘积求和改为一次矩阵乘法。实际速度取决于规模，需测量。

## 4. 伯努利：未出现项并入偏置

按照相同阈值二值化，严格使用 `X > binarize`；改变阈值不属于等价优化。平滑出现概率为：

$$
\theta_{cj}=\frac{N_{cj}^{(1)}+a}{N_c+2a}.
$$

使用 `BernoulliNB._update_feature_log_prob` 的对数之差生成参数。评分变形为：

$$
\begin{aligned}
L_{ic}
&=\log\pi_c+\sum_j\left[x_{ij}\log\theta_{cj}+(1-x_{ij})\log(1-\theta_{cj})\right]\\
&=b_c+\sum_jx_{ij}w_{cj},\\
w_{cj}&=\log\theta_{cj}-\log(1-\theta_{cj}),\\
b_c&=\log\pi_c+\sum_j\log(1-\theta_{cj}).
\end{aligned}
$$

参考 `BernoulliNB._joint_log_likelihood`，每次预测计算：

```python
neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
scores = X @ (self.feature_log_prob_ - neg_prob).T
scores += self.class_log_prior_ + neg_prob.sum(axis=1)
```

**收益：** 未出现特征的证据仍完整参与评分，避免逐类显式构造 `1-X` 和两组乘积，合并为矩阵评分。没有新增训练时权重缓存，也没有自行替换源码的未出现概率计算式。概率接近 1 时仍可能舍入并产生无穷对数。

## 5. 类别：直方图与批量查表

每个特征的编码范围大小为 $K_j=\max(1+\max_i x_{ij},K_{\min})$，其中 $K_{\min}$ 是标量 `min_categories` 的下限，未指定时取 1。源码对应 `CategoricalNB._validate_n_categories`。

`CategoricalNB._count` 按特征和类别生成掩码，用 `np.bincount` 统计，再更新非零计数。本实现保留这一方式。基础版已有直方图统计，不能将它算作新优化。每个特征的参数表形状为 `(C, K_j)`，编码上限决定表大小，不等于实际出现的不同取值数。

概率改用 `CategoricalNB._update_feature_log_prob` 的对数之差。预测按特征查出全部类别：

```python
scores = np.zeros((len(X), len(self.classes_)))
for j, log_prob in enumerate(self.feature_log_prob_):
    scores += log_prob[:, X[:, j]].T
scores += self.class_log_prior_
```

**收益：** 将基础版类别循环中的特征循环改为单层特征循环，计算规模仍为 $O(mdC)$。这是 `CategoricalNB._joint_log_likelihood` 的计算方式。

digits 使用 `min_categories=17`，预留 0～16 编码；训练中未出现的等级仍通过平滑得到概率。预测越界报错。只支持一次性训练，不维护跨批次编码范围。

## 6. 高斯：分项评分与统计量合并

继续采用中心化的最大似然方差，即 `ddof=0`。不改用 $E[X^2]-E[X]^2$，以免大均值、小方差时发生相消。

定义 `var_smoothing` 为 $s$，原始类内方差为 $v_{cj}$：

$$
\epsilon=s\max_j\mathrm{Var}(X_j),\qquad \widetilde v_{cj}=v_{cj}+\epsilon.
$$

该平滑基础版已实现，沿用 `GaussianNB._partial_fit` 在一次性训练中的计算式，不算新增优化。全常数数据可能仍有零方差，本实现保留基础版的正方差检查，没有加入固定下限。

### 分项评分

参考 `GaussianNB._joint_log_likelihood`：

$$
L_{ic}=\log\pi_c-\frac12\sum_j\log(2\pi\widetilde v_{cj})
-\frac12\sum_j\frac{(x_{ij}-\mu_{cj})^2}{\widetilde v_{cj}}.
$$

固定对数项先单独求和，偏差项按样本求和；基础版先将固定项广播加到每个样本的特征项后再求和。这是新增的等价计算结构。逐类预测本来就存在，不能将避免三维张量作为新增收益。

按源码在预测时计算对数先验、固定项和除法，不缓存高斯对数先验、固定项或方差倒数。复杂度仍为 $O(mdC)$。

### 均值方差合并公式

保留 `GaussianNB._update_mean_variance`，两组统计量为 $n_a,\mu_a,v_a$ 和 $n_b,\mu_b,v_b$，逐特征合并：

$$
n=n_a+n_b,\qquad \mu=\frac{n_a\mu_a+n_b\mu_b}{n},
$$

$$
M_2=n_av_a+n_bv_b+\frac{n_an_b}{n}(\mu_a-\mu_b)^2,\qquad v=\frac{M_2}{n}.
$$

其中 $M_2=\sum_i(x_i-\mu)^2$，均值差修正项不可省略。旧计数为零和空批次按源码分别处理。

**收益与条件：** 合并统计量时无需重新扫描旧数据，属于数学方法。一次性 `fit` 使用旧计数为零的分支，不因此获得分批训练能力或必然加速；非零旧计数分支由数学测试按不同分块验证。本阶段没有 `partial_fit`，也没有跨批次方差平滑处理。

## 7. 补集：直接生成权重

保留源码和基础版的总计数减本类计数：

$$
T_j=\sum_cN_{cj},\qquad \bar N_{cj}=T_j+a-N_{cj},
$$

$$
w_{cj}=-\log\frac{\bar N_{cj}}{\sum_\ell\bar N_{c\ell}},\qquad S=XW^\top.
$$

对应 `ComplementNB._count`、`_update_feature_log_prob` 和 `_joint_log_likelihood`。源码先求补集比例再取对数，本实现保持此顺序，不自行改为对数之差。

**收益：** 基础版先生成多项式条件概率，再覆盖为补集权重；分离计数与参数更新后，直接生成补集权重，避免无用计算。预测改为矩阵评分。默认 `norm=False`，至少两类，得分不加先验。

### 可选补集归一化

保留源码 `norm=True` 的数学选项：

$$
\widetilde w_{cj}=\frac{w_{cj}}{\sum_\ell w_{c\ell}}.
$$

它可能改变分类规则，单独与同参数 sklearn 比较，不当作等价性能优化。单特征等导致权重和为零时，本实现抛出 `ValueError`；这是计算定义域检查，不是源码新增优化，也不复现 sklearn 的全部无效值行为。

## 8. 数值保护与稳定归一化

### 可选的小平滑值下限

四种离散模型提供 `force_alpha=True`，与当前 sklearn 默认一致。定义构造参数 `alpha` 为 $\alpha$，实际计算值为 $a$：

$$
a=\begin{cases}
\alpha, & \text{保持输入平滑值},\\
\max(\alpha,10^{-10}), & \text{启用源码的数值下限}.
\end{cases}
$$

`force_alpha=False` 启用第二行；当输入小于下限时发出 `UserWarning`。`_BaseDiscreteNB._check_alpha` 返回有效值供参数更新，不修改模型的 `alpha` 属性。默认路径不截断，不能无条件加固定下限并声称与 sklearn 一致。

截断本身是数值保护，警告是接口行为；这里保留与截断直接关联的提示，完整参数和警告兼容留给第三阶段。本阶段仍限于有限正标量，不扩展为零或向量平滑。

### 对数概率归一化

`_BaseNB.predict_log_proba` 对每行得分 $L$ 使用稳定的 `logsumexp`：

$$
\log p_{ic}=L_{ic}-\log\sum_{k=1}^{C}\exp(L_{ik}),\qquad p_{ic}=\exp(\log p_{ic}).
$$

其稳定形式先减去每行最大值，避免直接指数的上溢或整体下溢。本实现使用 `scipy.special.logsumexp`，对应源码 `_logsumexp` 的计算方法。预测标签只取 `argmax`，不做无用的概率归一化；这一规则基础版已有。

提供概率输出是接口安排，采用稳定归一化才是数值方法。补集输出为归一化分类得分，不能解释为普通生成模型后验或已校准概率。

## 9. 验证与计时

三方演示使用与 `NB/main.py` 相同的 digits：整数灰度、`test_size=0.3`、`random_state=42`、`stratify=y`。参数为高斯 `var_smoothing=1e-9`，四种离散模型 `alpha=1.0`，类别 `min_categories=17`，伯努利 `binarize=8`，补集默认 `norm=False`。

[test_nb.py](test_nb.py) 有 13 项测试，覆盖手算示例、三方参数和评分对照、随机评分等价变形、不连续或字符串标签、平局顺序、概率稳定性、编码和阈值边界、高斯退化及大偏移小方差、均值方差分块合并、重复拟合重置、补集归一化、平滑值数值保护，以及稠密输入的阶段边界。

计数和标签精确比较，通常浮点参数、评分和概率使用 `rtol=1e-8, atol=1e-10`。大偏移小方差的统计量合并使用均值绝对误差 $2\times10^{-7}$、方差相对误差 $2\times10^{-6}$。近似平局比较得分并检查本模型的 `argmax` 规则，不要求不同运算顺序逐位相同。

模型和 sklearn 分别处理原始输入，对照 sklearn 的公共 `predict_joint_log_proba`，避免把本模型的预处理结果传给参考实现而掩盖差异。CSR 包括重复索引均拒绝处理，不会自行先求和再二值化。

[benchmark.py](benchmark.py) 只测相同稠密输入上的训练与预测，包含 digits 和两种合成规模，共 15 组模型记录。计时包含基本检查和计算，排除数据准备、模型构造及打印；预热后重复测量并取中位数。线程限制只用于控制测量条件，不是模型的资源优化。计数矩阵、输出数组的大小可用于理解方法成本，但本基准不测资源优化或峰值内存。

在项目根目录运行：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.optimized.main
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m unittest NB.optimized.test_nb NB.basic.test_nb
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.optimized.benchmark --repeats 7 --threads 1 --output NB/optimized/benchmark_results.json
```

## 10. 当前代码的实测结果

重新测量于 2026-09-30，环境为 Python 3.13.15、NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1，Linux x86_64，BLAS/OpenMP 单线程。预热一次、重复七次，下面为中位数，耗时单位为毫秒；未舍入数据和 sklearn 耗时见 [benchmark_results.json](benchmark_results.json)。记录只反映该次测量条件。

### digits

训练集 1257 个样本，测试集 540 个样本，64 个特征、10 个类别。三方预测一致；优化版评分与 basic、sklearn 在设定误差内一致，概率与 sklearn 在设定误差内一致。

| 模型 | 三方准确率 | basic fit | optimized fit | basic predict | optimized predict | 预测耗时比 basic/optimized |
|---|---:|---:|---:|---:|---:|---:|
| GaussianNB | 82.22% | 0.318 | 0.329 | 0.677 | 0.479 | 1.41 |
| MultinomialNB | 87.59% | 0.106 | 0.115 | 0.274 | 0.033 | 8.18 |
| CategoricalNB | 90.00% | 2.363 | 4.181 | 1.053 | 0.338 | 3.12 |
| BernoulliNB | 87.41% | 0.129 | 0.132 | 0.589 | 0.050 | 11.68 |
| ComplementNB | 79.44% | 0.114 | 0.115 | 0.266 | 0.030 | 8.75 |

此规模预测耗时下降，训练没有统一收益。类别训练采用源码掩码和非零直方图更新，反而慢于基础版；不能因采用源码形式就宣称训练优化成功。高斯评分收益来自固定项与偏差项分别求和的结构，没有新增三维张量规避或缓存。

### 合成稠密数据

特征为 0～16 的均匀随机整数，以 `float64` 保存，类别均衡后随机打乱，种子为 42，参数沿用 digits。规模为 $(n,d,C)$，预测样本数为 $n/3$。下表依次给出训练、预测耗时比 basic/optimized，大于 1 表示优化版耗时更少。

| 模型 | $(600,32,3)$ fit / predict | $(2400,128,10)$ fit / predict |
|---|---:|---:|
| GaussianNB | 0.93 / 1.16 | 1.06 / 1.38 |
| MultinomialNB | 1.07 / 3.37 | 0.79 / 7.84 |
| CategoricalNB | 0.49 / 1.47 | 0.64 / 2.82 |
| BernoulliNB | 1.17 / 3.88 | 0.83 / 10.04 |
| ComplementNB | 1.19 / 3.60 | 0.80 / 7.01 |

15 组基准的三方预测一致，优化版评分及概率均与 sklearn 在误差内一致。预测收益随规模变化，训练有升有降，类别训练在这些规模均更慢，不宣称矩阵计数总能加速。本目录和基础版合并运行共 19 项测试通过。
