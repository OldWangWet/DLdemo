# 优化版朴素贝叶斯：实现与验证

本目录对应[项目 README](../../readme.md) 的第二阶段：在基础实现上，参考 sklearn 源码进行算法和数学优化，解释依据、收益与适用条件，并对照验证结果。五种模型、三方演示、正确性测试和性能基准均已实现；增量训练、三种模型的 CSR 输入及补集权重归一化也已验证。代码入口为 [nb.py](nb.py)，实测记录见第 10 节。

原理以 [NB README](../readme.md) 为准，现有代码以 [基础实现](../basic/nb.py) 为起点。已核对 `DLdemo` 环境中的 scikit-learn 1.9.1，源码位于 `/home/fire/miniconda3/envs/DLdemo/lib/python3.13/site-packages/sklearn/naive_bayes.py`。本文只列该版本源码已经实现的方法，每项均给出对应源码位置；代码片段仅简化数组后端和接口细节，计算方式与源码一致。

## 1. 本阶段的目标与边界

核心目标是让五种模型的公式变成可复用的批量计算流程。对相同数据和参数，等价变形应保留分类规则；浮点计算顺序改变后，参数和得分允许有合理误差。优化主要改善计算成本、重复计算和数值稳定性，不保证提高准确率。

**核心实现：** 保留 `fit(X, y)`、`predict(X)`，新增批量得分与概率输出；采用源码中的离散计数和矩阵预测、伯努利得分变形、类别批量查表及高斯均值方差合并方法。

**已实现扩展：** 五种模型支持 `partial_fit`；多项式、伯努利和补集支持 CSR；补集支持 `norm=True`。这些能力采用源码已有计算方式，接口边界见第 8 节。

**第三阶段：** `sklearn_impl/` 负责完整接口和行为复现，包括 estimator 协议、参数管理、全面输入校验、样本权重、完整稀疏与增量接口、异常和警告兼容，以及进一步的 CPU、内存优化。本阶段模型计算使用 NumPy，概率归一化可使用 SciPy；sklearn 仅用于数据加载和对照验证，不调用其模型完成训练或预测。

### 基础版已经做了什么

| 能力 | `basic` 现状 | `optimized` 的实际增量 |
|---|---|---|
| 对数域计算 | 五种模型均已采用 | 统一批量得分接口，补充稳定的概率归一化 |
| 离散概率缓存 | 已保存 `feature_log_prob_` | 合并所有类别的预测计算，减少逐类临时数组 |
| 方差与计数平滑 | 已有高斯平滑和正数 `alpha` | 明确退化边界，保持相同参数语义 |
| 类别直方图 | 已使用 `np.bincount` 和逐特征参数表 | 保留源码的逐组直方图，一次查出所有类别的贡献 |
| 补集计数 | 已用总计数减本类计数 | 避免先计算随后被覆盖的多项式概率，复用矩阵预测 |
| 高斯预测与统计 | 已按类别处理，未创建三维张量 | 保留逐类预测结构；新增源码中的均值方差合并方法 |

上述基础能力继续保留；评估优化收益时，要与现有 `basic` 比较，不能把已经实现的能力当作新增收益。

### 优化方法与源码对应关系

| 方法 | sklearn 源码位置 |
|---|---|
| 标签预测直接取最大得分，概率按需归一化 | `_BaseNB.predict`、`predict_log_proba`、`predict_proba` |
| 离散标签指示矩阵、计数与参数更新分离 | `_BaseDiscreteNB.fit`、`partial_fit` |
| 多项式矩阵计数、对数之差、矩阵预测 | `MultinomialNB._count`、`_update_feature_log_prob`、`_joint_log_likelihood` |
| 伯努利未出现项变形 | `BernoulliNB._joint_log_likelihood` |
| 类别直方图、独立大小的参数表、批量查表 | `CategoricalNB._count`、`_validate_n_categories`、`_joint_log_likelihood` |
| 高斯均值方差合并、方差平滑、逐类预测 | `GaussianNB._update_mean_variance`、`_partial_fit`、`_joint_log_likelihood` |
| 补集计数、直接生成补集权重及可选归一化 | `ComplementNB._count`、`_update_feature_log_prob` |
| CSR 矩阵乘法 | 上述支持稀疏输入模型中的 `safe_sparse_dot`，实现位于 `sklearn/utils/extmath.py` |

文件组织、实施顺序和测试方案是本项目的安排；它们不作为 sklearn 的算法优化项。

## 2. 文件结构与公共计算流程

文件结构与基础目录对应：

```text
NB/optimized/
├── __init__.py       # 导出五个模型
├── nb.py             # 公共流程、统计量和各模型计算
├── main.py           # digits 上对照 basic、optimized、sklearn
├── test_nb.py        # 数学等价性和结果验证
├── benchmark.py      # 分开测量训练与预测耗时
├── benchmark_results.json  # 实测耗时、环境、参数和数组尺寸
└── readme.md
```

记训练样本数为 $n$、预测样本数为 $m$、特征数为 $d$、类别数为 $C$，类别特征 $j$ 的编码范围大小为 $K_j$。`_BaseDiscreteNB.fit` 使用 `LabelBinarizer` 构造标签指示矩阵 $Y$，保存排序后的 `classes_`；二分类时将单列输出补成两列，单类别时使用全 1 列。本阶段手动实现相同的指示矩阵语义。

`classes_` 决定所有参数表和概率输出的类别顺序。样本标签可以是不连续整数或字符串，不直接把原始标签当作数组下标。普通四种模型根据类别计数计算经验先验；补集模型保留计数供统计使用，但沿用基础版至少两个类别、预测不加先验的规则。

公共基类负责输入检查、拟合状态、特征数检查、类别映射以及输出转换。公共 `joint_log_likelihood(X)` 校验输入后调用各模型的 `_joint_log_likelihood(X)`，返回形状为 `(m, C)` 的得分矩阵；补集模型的方法名沿用 sklearn，但实际返回分类得分。

```python
def predict(self, X):
    scores = self.joint_log_likelihood(X)
    return self.classes_[scores.argmax(axis=1)]
```

参考 `_BaseNB.predict` 的流程，在公共预测入口检查拟合状态和输入，再调用模型得分方法。本阶段要求有限数值二维数据、非空训练集、一维标签和有限标量 `alpha > 0`；高斯、类别仅支持稠密输入，其余三种模型另支持 CSR。`alpha` 不做额外截断；完整输入与异常行为留给第三阶段。

五种模型的公共接口为 `fit(X, y)`、`partial_fit(X, y, classes=None)`、`joint_log_likelihood(X)`、`predict(X)`、`predict_log_proba(X)` 和 `predict_proba(X)`；训练方法返回模型自身。构造参数与第 9 节对照表一致，另支持补集 `norm=True` 和伯努利 `binarize=None`。

```python
from NB.optimized import MultinomialNB

model = MultinomialNB(alpha=1.0).fit([[3, 1], [0, 4]], ["垃圾", "正常"])
labels = model.predict([[2, 0]])
scores = model.joint_log_likelihood([[2, 0]])
proba = model.predict_proba([[2, 0]])  # 列顺序由 model.classes_ 决定
```

### 按需计算概率

`_BaseNB.predict_log_proba` 使用 `_logsumexp` 归一化；NumPy/SciPy 实现可使用同一计算方法：

$$
\log p_{ic}=L_{ic}-\log\sum_{k=1}^{C}\exp(L_{ik}),
\qquad
p_{ic}=\exp(\log p_{ic}).
$$

```python
from scipy.special import logsumexp

log_proba = scores - logsumexp(scores, axis=1, keepdims=True)
proba = np.exp(log_proba)
```

其稳定形式先减去每行最大值，再求指数和，避免直接 `np.exp(scores)` 的上溢或整体下溢。仅预测类别时直接 `argmax`，无需做概率归一化。对于补集模型，这两个接口输出归一化得分，不能把它解释成普通朴素贝叶斯生成模型的后验概率，也不能据此声称概率已校准。

## 3. 多项式：共享计数与矩阵预测

### 训练计数

将标签编码为形状 $(n,C)$ 的类别指示矩阵 $Y$，每行在所属类别位置取 1，其余位置取 0。类别特征计数表为：

$$
N=Y^\top X,\qquad N\in\mathbb R^{C\times d}.
$$

```python
# Y 为公共训练流程生成的类别指示矩阵。
self.feature_count_ += Y.T @ X
self.class_count_ += Y.sum(axis=0)
```

这是 `MultinomialNB._count` 的稠密版本，伯努利及补集源码也采用相同的矩阵计数。它减少 Python 层逐类筛选和求和，但密集乘法仍有约 $O(nCd)$ 运算及 $O(nC)$ 的指示矩阵开销。基础版按类求和的有效特征处理量约为 $O(nd)$，另有重复掩码检查；因此不能宣称矩阵计数必然更快或复杂度更低。

### 平滑与预测

标量平滑参数为 $\alpha$，保存：

$$
\log\theta_{cj}=\log(N_{cj}+\alpha)
-\log\sum_{\ell=1}^{d}(N_{c\ell}+\alpha).
$$

采用“对数之差”而不是先求概率比值再取对数，与 `MultinomialNB._update_feature_log_prob` 一致，可避免概率比值过小时的部分下溢问题。计数求和本身仍需处于浮点可表示范围。

预测一次计算全部类别：

$$
L=X(\log\Theta)^\top+\log\boldsymbol\pi.
$$

```python
scores = X @ self.feature_log_prob_.T
scores += self.class_log_prior_[None, :]
```

**收益与条件：** 预测仍为 $O(mdC)$，但可使用 NumPy 底层矩阵运算，避免逐类别构造 `(m, d)` 的乘积数组。重复批量预测通常更适合这种形式，实际收益需测量。灰度输入继续按计数式权重使用，与现有示例一致。

## 4. 伯努利：把未出现贡献并入偏置

基础版对每个类别显式计算 $X\log\theta_c+(1-X)\log(1-\theta_c)$。利用代数变形：

$$
\begin{aligned}
L_{ic}
&=\log\pi_c+\sum_j\left[x_{ij}\log\theta_{cj}
+(1-x_{ij})\log(1-\theta_{cj})\right]\\
&=b_c+\sum_jx_{ij}w_{cj},\\
w_{cj}&=\log\theta_{cj}-\log(1-\theta_{cj}),\\
b_c&=\log\pi_c+\sum_j\log(1-\theta_{cj}).
\end{aligned}
$$

训练时先按相同阈值二值化，以共享计数流程统计特征出现次数，再计算：

$$
\theta_{cj}=\frac{N_{cj}^{(1)}+\alpha}{N_c+2\alpha}.
$$

源码在每次预测时计算未出现概率的对数、权重差和偏置，再执行矩阵乘法：

```python
neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
scores = X_binary @ (self.feature_log_prob_ - neg_prob).T
scores += self.class_log_prior_ + neg_prob.sum(axis=1)
```

上述 `neg_prob` 的计算方式与当前源码一致；概率非常接近 1 时仍可能遇到浮点舍入和无穷对数，不能声称这项代数变形消除了所有数值问题。

**收益：** 特征未出现的证据仍完整参与评分；不再为每个类别构造 `1-X` 和两组逐元素乘积，且为 CSR 扩展提供直接的矩阵形式。

**源码位置：** `BernoulliNB._count`、`_update_feature_log_prob`、`_joint_log_likelihood`。本阶段按源码在预测时计算权重差和偏置。

保留 `binarize=8` 时严格使用 `X > 8`；`binarize=None` 时要求输入已经是 0/1。不要把阈值或分布模型的改变算作等价优化。

## 5. 类别：直方图统计与批量查表

### 训练

`CategoricalNB._count` 按特征和类别构造掩码，调用 `np.bincount` 统计取值次数；基础版已有这一计算方式。每个特征单独确定：

$$
K_j=\max\left(1+\max_i x_{ij},K_{\min}\right),
$$

其中 $K_{\min}$ 表示标量 `min_categories` 给出的下限；未指定时取 1。计数表和对数概率表用列表保存，每项形状为 `(C, K_j)`，参数存储为 $O(C\sum_jK_j)$。

编码必须是非负整数。表大小取决于最大编码加一，而非实际出现的不同取值数；不连续的大编码会浪费空间。这是源码存储方式的适用边界。

### 预测

参考 `CategoricalNB._joint_log_likelihood`，按特征查出全部类别的贡献，再加先验：

```python
scores = np.zeros((len(X), len(self.classes_)))
for j, log_prob in enumerate(self.feature_log_prob_):
    scores += log_prob[:, X[:, j]].T
scores += self.class_log_prior_
```

**收益：** 直接从参数表读取 `(m, C)` 的贡献，把基础版“类别循环内再按特征循环”变为单层特征循环。计算规模仍为 $O(mdC)$；不需要把输入展开为完整 one-hot 特征矩阵。参考 `CategoricalNB._joint_log_likelihood`。

digits 继续设置 `min_categories=17`，训练未出现但位于 0～16 范围内的等级通过平滑获得概率；超过拟合参数表范围的预测编码报错。增量训练遇到更大编码时按源码使用 `np.pad` 扩展计数表，保留旧计数；后续小编码批次不缩小参数表。

## 6. 高斯：逐类预测与稳定统计量合并

### 保留中心化方差和方差平滑

每类按最大似然方差计算，即 `ddof=0`，继续使用中心化偏差或 `np.var`。不要为了矩阵化改用 $E[X^2]-E[X]^2$：特征均值很大、方差很小时，两个大数相减可能严重丢失精度。

定义 `var_smoothing` 为 $s$，原始类内方差为 $v_{cj}$，预测使用的平滑方差为 $\widetilde v_{cj}$：

$$
\epsilon=s\max_j\mathrm{Var}(X_j),\qquad
\widetilde v_{cj}=v_{cj}+\epsilon.
$$

这与基础版和一次性 sklearn `fit` 保持一致。全体特征均为常数时，$\epsilon$ 可能仍为零；沿用基础版的严格正方差检查，不额外添加未说明的固定下限，以免改变模型语义。

### 逐类别计算预测得分

`GaussianNB._joint_log_likelihood` 每次预测时按类别计算：

$$
L_{ic}=\log\pi_c-\frac12\sum_j\log(2\pi\widetilde v_{cj})
-\frac12\sum_j\frac{(x_{ij}-\mu_{cj})^2}{\widetilde v_{cj}}.
$$

各类别的结果加入列表，最后堆叠成 `(m, C)` 的得分矩阵。源码在预测时计算对数项和除法，未保存高斯固定项或倒数缓存。

**收益与边界：** 逐类偏差计算的主要临时空间为 $O(md)$，避免创建 `(m, C, d)` 张量，预测复杂度仍为 $O(mdC)$。基础版已经采用逐类计算，这部分作为沿用的源码方法，不能算成新增优化。

### 可合并的统计量

采用 `GaussianNB._update_mean_variance` 的均值方差合并方法。两组同类数据的样本数、均值和方差分别为 $n_a,\mu_a,v_a$ 与 $n_b,\mu_b,v_b$，逐特征计算：

$$
n=n_a+n_b,\qquad
\mu=\frac{n_a\mu_a+n_b\mu_b}{n},
$$

$$
M_2=n_av_a+n_bv_b+\frac{n_an_b}{n}(\mu_a-\mu_b)^2,\qquad
v=\frac{M_2}{n}.
$$

其中 $M_2=\sum_i(x_i-\mu)^2$。均值之间的修正项不可省略，不能直接平均两批方差；空批次和旧计数为零的情形分别处理。

**收益与条件：** 只保存 $O(Cd)$ 的类内统计量即可合并不同批次，适合分批训练。`fit` 与 `partial_fit` 共享该函数；测试另外对未平滑统计量按 1、2、7、101 块合并并验证。

增量接口沿用 `GaussianNB._partial_fit` 的平滑处理顺序：每次根据当前批次计算并覆盖 `epsilon_`，后续批次先从已有方差减去这个**当前批次**的平滑量，合并统计量后再加回；这里没有改成减去上一批次的平滑量。源码未维护累计全局方差来决定平滑量，因此不能保证任意分批训练与一次性 `fit` 得到相同方差。测试使用变化的批次分布和 `var_smoothing=0.01`，按相同批次、相同顺序与 sklearn 对照。

## 7. 补集：共享原始计数，直接构造权重

基础版已经使用“全部计数减本类计数”，优化版继续保留：

$$
T_j=\sum_cN_{cj},\qquad
\bar N_{cj}=T_j-N_{cj}+\alpha,
$$

$$
w_{cj}=-\log\frac{\bar N_{cj}}{\sum_\ell\bar N_{c\ell}},\qquad
S=XW^\top.
$$

参考 `ComplementNB._update_feature_log_prob` 和 `_joint_log_likelihood`。源码的具体计算为：

```python
comp_count = self.feature_all_ + alpha - self.feature_count_
logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))
self.feature_log_prob_ = -logged  # norm=False
```

这里源码先计算比值再取对数，与多项式源码采用的对数之差有所不同，本阶段保留其实际方式。

`_BaseDiscreteNB.fit` 先调用 `_count`，再调用各模型的 `_update_feature_log_prob`；`ComplementNB` 在自己的参数更新方法中直接生成补集权重。基础版调用 `MultinomialNB._fit` 会先计算本类的平滑概率，再覆盖为补集权重；按源码的计数、参数更新分离流程可避免这部分无用计算。

**收益：** 得到计数后，用 $O(Cd)$ 的计算构造全部补集，无需为每类复制补集训练数据；预测改为一次矩阵乘法。本阶段默认保持 `norm=False`、至少两个类别，得分不加类别先验。

已支持 `norm=True`，在权重有限且每行和非零时令：

$$
\widetilde w_{cj}=\frac{w_{cj}}{\sum_\ell w_{c\ell}}.
$$

这是另一种分类规则，可能改变预测，已单独与相同设置的 sklearn 比较；不能作为“保持结果不变”的性能优化。单特征等导致权重和为零时，本阶段明确抛出 `ValueError`，不生成无效归一化权重。

## 8. 增量和稀疏扩展的实现方式

| 已实现扩展 | 实现方式 | 适用条件与边界 |
|---|---|---|
| 离散增量训练 | 按 `_BaseDiscreteNB.partial_fit` 累加原始计数，再更新对数概率和先验 | 分批读取大数据；每批重新生成参数存在固定开销 |
| 高斯增量训练 | 按 `GaussianNB._partial_fit` 合并类内计数、均值和方差，使用当前批次的平滑量 | 避免保存历史样本；按相同分批顺序对照源码 |
| 多项式、补集 CSR | 使用 SciPy CSR 和稀疏矩阵乘法，避免 `np.asarray` 或 `toarray` | 非负高维稀疏输入；参数表和得分输出仍为稠密 |
| 伯努利 CSR | 保留未出现偏置，仅对非零项二值化并进行矩阵乘法 | 稀疏阈值应非负，确保隐式零仍被映射为 0；`binarize=None` 校验非零项均为 1 |

参考源码已有的 `partial_fit(X, y, classes=None)` 接口：首次传入全部标签，后续批次可以缺少部分类别，类别集合保持固定；`fit` 则重新初始化统计量。未观察到的类别计数为零，经验对数先验为负无穷，已验证这类中间批次的参数和概率与源码一致。

CSR 在校验时复制并合并重复索引，伯努利仅处理存储项，不修改调用者的输入；训练和预测均直接使用矩阵乘法，不调用 `toarray`。测试覆盖 `csr_matrix` 和 `csr_array`，包括 CSR 增量训练，并禁止在模型流程中调用 `toarray`。

**接口差异：** 本阶段保留基础版更严格的范围：类别输入必须为非负整数，伯努利 `binarize=None` 必须为 0/1，补集至少两类，归一化补集拒绝零权重和，高斯最终方差必须有限且严格大于零。因而全常数的首次训练批次、零平滑时未观察到的高斯类别等退化情况会报错；这不复现 sklearn 的全部警告和无效值行为。类别模型的 `n_categories_` 记录累计参数表大小；sklearn 1.9.1 的该属性按当前批次重新计算，但实际计数表仍保留历史大小。测试对照累计计数表、概率表及预测，不把该属性的中间批次差异视为算法结果差异。

稀疏矩阵预测的主要乘法工作量约为 $O(\mathrm{nnz}(X)C)$，此外仍需存储 `(m, C)` 输出。仅在输入足够稀疏时才可能获益；高斯和类别模型本阶段保持稠密输入。性能基准另包含密度为 1% 和 10% 的合成数据，比较同一优化模型的稠密与 CSR 路径。

离散平滑保持有限标量 `alpha > 0`。零平滑、向量 `alpha` 以及 `force_alpha` 的警告兼容留给第三阶段；sklearn 当前默认 `force_alpha=True`，不会自动把所有过小值截断，因此不能无说明地加入固定下限并声称与其默认行为一致。

## 9. 正确性验证与性能评估

### 对照数据和参数

`main.py` 使用与 [NB/main.py](../main.py) 完全相同的 digits 数据：转为整数、`test_size=0.3`、`random_state=42`、`stratify=y`。三种实现使用相同训练和测试集：

| 模型 | 对照参数 | 必须比较的内容 |
|---|---|---|
| 高斯 | `var_smoothing=1e-9` | 类别计数、均值、平滑量、方差、得分和预测 |
| 多项式 | `alpha=1.0` | 原始计数、先验、条件对数概率、得分和预测 |
| 类别 | `alpha=1.0, min_categories=17` | 各特征编码范围、计数表、概率表和预测 |
| 伯努利 | `alpha=1.0, binarize=8` | 二值化结果、出现次数、概率表、原始与变形后的得分 |
| 补集 | `alpha=1.0, norm=False` | 原始计数、补集权重、无先验得分和预测 |

准确率用于确认示例表现，不能单独证明数学实现正确。`classes_` 及原始整数计数应精确一致；浮点参数、得分和概率用 `assert_allclose`，例如从 `rtol=1e-8, atol=1e-10` 开始，结合数据量和尺度解释误差。digits 上检查预测一致；构造近似平局数据时，应检查得分误差并保留按 `classes_` 顺序取首个最大值的规则，不要求不同运算顺序逐位一致。

### 有意义的验证用例

1. 复用上层 README 的五个手算示例，保证优化后仍符合公式。
2. 用随机小矩阵直接比较多项式矩阵得分、伯努利原始得分与变形得分、类别逐类查表与批量查表，覆盖全零特征和不连续标签。
3. 检查概率每行和为 1、概率与对数概率相符，以及极负得分下稳定归一化仍能输出有限结果。
4. 检查类别预留等级、编码越界、伯努利阈值恰好等于 8、`binarize=None` 的非二元输入、特征数不符及未拟合预测。
5. 检查高斯恒定像素、大偏移小方差、单样本类别；用相同类别数据按不同分块合并统计量，与一次性均值及 `ddof=0` 方差比较。
6. 验证离散一次性与分批统计、稠密与 CSR 结果，以及补集 `norm=True` 与同参数 sklearn 的一致性。高斯增量与相同批次及顺序的 sklearn 对照。

[test_nb.py](test_nb.py) 的 16 项测试覆盖上述用例。通常使用 `rtol=1e-8, atol=1e-10`；大偏移小方差用例的特征均值约为 $10^8$、标准差约为 0.2，分块合并受浮点舍入影响，均值使用绝对误差 $2\times10^{-7}$，方差使用相对误差 $2\times10^{-6}$。近似平局检查得分误差及本模型的 `argmax` 规则，完全平局返回排序后首个类别。

### 性能评估

`benchmark.py` 分别报告 `fit` 和 `predict` 耗时，测量批量矩阵计算的实际效果。使用相同数据、参数和 dtype，提前准备数据，排除加载、划分及打印开销；先预热，再多次计时并报告中位数。记录 NumPy、sklearn 版本和 BLAS 线程设置，避免把不同线程数造成的变化归因于公式优化。

数据包括 digits 及不同 $n,d,C$ 的合成数据；稀疏扩展增加不同稀疏度的数据。记录输入、参数及主要临时数组的规模；若测进程峰值内存，应单独运行各实现，不能只用 `tracemalloc` 推断全部 NumPy 原生分配。报告优化前后耗时和比值，允许某些规模没有加速；没有实测前不填写性能提升百分比。

在项目根目录执行以下命令：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.optimized.main
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m unittest NB.optimized.test_nb
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.optimized.benchmark
```

保存完整性能记录：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.optimized.benchmark --repeats 7 --threads 1 --output NB/optimized/benchmark_results.json
```

`benchmark.py` 默认预热一次、重复五次，用 `threadpoolctl` 将 BLAS/OpenMP 线程数限制为 1；可通过 `--repeats` 和 `--threads` 调整。计时包含模型校验和训练/预测计算，不包含模型构造、数据准备、CSR 转换和输出。稠密三方使用相同 `float64` 输入，类别模型内部再转为整数；CSR 对照使用同一矩阵的不同表示。JSON 保存 sklearn 耗时、全部计时样本、版本、线程设置、输入字节数、模型数组字节数、标签指示矩阵和主要预测临时数组的尺寸估算；这些字节数不是进程峰值，也不表示全部临时内存。

## 10. 实测结果与适用条件

记录日期为 2026-09-30，环境为 Python 3.13.15、NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1，Linux x86_64，OpenBLAS 单线程。下面取预热一次、重复七次的中位数，耗时单位为毫秒；完整环境及未舍入数据见 [benchmark_results.json](benchmark_results.json)。短任务的耗时会随机器负载变化，这些记录不保证在其他环境获得相同比值。

### digits 正确性与耗时

训练集 1257 个样本，测试集 540 个样本，64 个特征、10 个类别。五种模型的三方预测完全一致，优化版得分与基础版、sklearn 在设定误差内一致，概率与 sklearn 在设定误差内一致。

| 模型 | 三方准确率 | basic fit | optimized fit | basic predict | optimized predict | 预测耗时比 basic/optimized |
|---|---:|---:|---:|---:|---:|---:|
| GaussianNB | 82.22% | 0.313 | 0.356 | 0.615 | 0.439 | 1.40 |
| MultinomialNB | 87.59% | 0.103 | 0.133 | 0.279 | 0.036 | 7.72 |
| CategoricalNB | 90.00% | 2.334 | 4.234 | 1.056 | 0.355 | 2.97 |
| BernoulliNB | 87.41% | 0.124 | 0.149 | 0.508 | 0.052 | 9.68 |
| ComplementNB | 79.44% | 0.110 | 0.133 | 0.220 | 0.033 | 6.74 |

这组数据上预测耗时下降，但训练耗时上升。矩阵计数需要构造标签指示矩阵，校验、统计与参数更新也有开销；类别模型沿用源码的掩码、直方图及非零索引更新，未把基础版已经具有的直方图统计列为新增加速。高斯模型继续逐类预测，其差别主要来自源码对固定对数项与偏差项分别求和的计算结构，收益不能归因于消除基础版并未使用的三维张量。

### 不同稠密规模

合成特征为 0～16 的均匀随机整数，输入保存为 `float64`，标签按类别均衡生成后打乱；参数沿用 digits。记规模为 $(n,d,C)$，预测样本数为 $n/3$。下表的两个数字依次为训练、预测耗时比，均为 basic/optimized；大于 1 表示优化版耗时更少。

| 模型 | $(600,32,3)$ fit / predict | $(2400,128,10)$ fit / predict |
|---|---:|---:|
| GaussianNB | 0.74 / 1.13 | 1.00 / 1.31 |
| MultinomialNB | 0.73 / 2.82 | 0.72 / 6.54 |
| CategoricalNB | 0.47 / 1.43 | 0.63 / 2.82 |
| BernoulliNB | 0.80 / 3.14 | 0.81 / 10.28 |
| ComplementNB | 0.74 / 2.83 | 0.74 / 6.66 |

预测收益随批量规模、特征数和类别数变化；这些样本未显示训练加速，不据此声称矩阵计数总能降低训练成本。五种模型在这些基准数据上的三方预测均一致。

### CSR 扩展

训练集 1800 个样本、400 个特征、6 个类别，预测集 600 个样本；非零值为 1～4，随机保留约 1% 或 10% 元素。伯努利使用 `binarize=0.0`，多项式和补集使用 `alpha=1.0`。下表的比值为 **optimized 稠密路径 / optimized CSR 路径**，不与基础版混用；同参数 sklearn CSR 的耗时保存在 JSON 中。

| 目标密度 | 模型 | fit 耗时比 | predict 耗时比 |
|---|---|---:|---:|
| 1% | MultinomialNB | 5.32 | 4.56 |
| 1% | BernoulliNB | 5.65 | 4.18 |
| 1% | ComplementNB | 5.01 | 4.76 |
| 10% | MultinomialNB | 3.08 | 2.23 |
| 10% | BernoulliNB | 2.97 | 2.18 |
| 10% | ComplementNB | 3.11 | 2.32 |

这些稀疏输入上 CSR 路径耗时更少，密度提高后收益下降。稠密与 CSR 的预测一致，概率在设定误差内一致；CSR 预测也与 sklearn 一致。计数表和概率表仍为稠密，预测输出仍需保存全部样本与类别的得分，稀疏输入不消除这些存储成本。

### 完成范围

五种手动模型、公共得分和概率接口、三方演示、16 项优化版测试及 21 组性能记录均已完成；扩展包含五种模型的增量训练、三种离散模型的 CSR 和补集归一化。与基础版和 KNN 的测试合并运行时，共 28 项测试通过。未实现样本权重、自定义先验、向量或零 `alpha`、estimator 协议及完整 sklearn 校验、警告和异常兼容；这些留给第三阶段。
