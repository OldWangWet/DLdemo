# KNN：从课本原理到高效近邻搜索

本目录以手动实现的方式呈现 sklearn 风格的 K 近邻分类器：从距离计算、近邻选择，到 KD-Tree、投票和概率输出，串起一条完整的实现路径。本文按本目录的实现讲解算法与优化，不进行 sklearn 源码一致性审查，也不讨论 CPython、Cython、编译扩展等语言级优化。

## 1. 课本上的 KNN 原理

### 1.1 基本思想

给定训练集

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^{n},\qquad x_i\in\mathbb R^d,
$$

对待预测样本 $x$，计算它与训练样本的距离，选出距离最小的 $k$ 个样本，记其索引集合为 $N_k(x)$。分类时，让这些近邻投票：

$$
\hat y(x)=\arg\max_c\sum_{i\in N_k(x)}\mathbf 1(y_i=c).
$$

例如，最近的 5 个邻居标签依次为 `A、B、A、A、B`，A 获得 3 票，预测类别就是 A。

KNN 是非参数、基于实例的学习方法。它没有梯度下降或反复更新模型参数的训练过程；`fit` 的主要工作是保存数据、编码类别，以及按需建立查询索引。计算的重心在预测阶段。

KNN 也可用于回归：将投票替换为邻居目标值的平均或加权平均即可。本目录实现的是分类器。

### 1.2 距离如何定义

Minkowski 距离为：

$$
D_p(x,z)=\left(\sum_{j=1}^{d}|x_j-z_j|^p\right)^{1/p}.
$$

| 名称 | 参数 | 公式 |
| --- | --- | --- |
| Manhattan / 曼哈顿距离 | $p=1$ | $\sum_j|x_j-z_j|$ |
| Euclidean / 欧氏距离 | $p=2$ | $\sqrt{\sum_j(x_j-z_j)^2}$ |
| Chebyshev / 切比雪夫距离 | $p=\infty$ | $\max_j|x_j-z_j|$ |
| 一般 Minkowski 距离 | 有限 $p\ge 1$ | 上述 $D_p$ |

$0<p<1$ 时仍可用相应表达式排序，但它不满足通常距离度量要求的三角不等式。本目录允许通过暴力搜索处理这种情况，不使用 KD-Tree。

距离对特征尺度敏感。如果一个特征以千为量级，另一个以小数为量级，前者可能主导距离。因此示例使用标准化：

$$
z_j=\frac{x_j-\mu_j}{\sigma_j}.
$$

均值和标准差只能从训练集估计，再用于转换测试集，避免泄漏测试集信息。

### 1.3 加权投票和概率

均匀投票令每个邻居的权重 $w_i=1$；距离投票令较近的样本有更大的影响：

$$
w_i=\frac{1}{D(x,x_i)},\qquad
S_c(x)=\sum_{i\in N_k(x)}w_i\mathbf 1(y_i=c).
$$

分类结果与概率分别为：

$$
\hat y(x)=\arg\max_c S_c(x),\qquad
P(y=c\mid x)=\frac{S_c(x)}{\sum_{i\in N_k(x)}w_i}.
$$

这里的概率是邻域内归一化的票数，不等同于经过专门校准的概率估计。

遇到距离为 0 的邻居时，不能直接用无穷大的倒数完成归一化。本目录对该查询只保留零距离邻居的投票：它们的权重置为 1，其余邻居置为 0。若多个完全重合的邻居标签不同，就在这些重合样本之间投票。

### 1.4 k 的作用与朴素算法的开销

较小的 $k$ 更关注局部结构，也更容易受噪声影响；较大的 $k$ 使边界更平滑，但可能忽略局部差异。多分类中即使 $k$ 是奇数，也不能保证没有平票。

对一个查询点，最直接的课本实现是：

1. 遍历全部 $n$ 个训练样本，计算距离：$O(nd)$。
2. 对全部距离排序：$O(n\log n)$。
3. 取前 $k$ 个邻居并投票。

设查询样本数为 $m$，完整距离矩阵需要 $O(mn)$ 空间。如何减少不必要的距离计算、全量排序和临时存储，就是后面实现的重点。

## 2. sklearn 的实现思路与本目录结构

sklearn 将近邻搜索与分类投票分开：搜索后端包括暴力搜索、KD-Tree、Ball Tree，并可通过 `auto` 根据数据条件选择。KD-Tree 使用轴对齐区域；Ball Tree 使用中心和半径描述区域。树通过距离界排除无须搜索的区域，而高维数据或较大的 $k$ 往往更适合暴力搜索。参见 [sklearn 官方近邻算法说明](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms)。

本目录围绕 **分块暴力搜索和 KD-Tree** 展开。后面的具体阈值、函数职责和数据流均以本目录代码为依据；Ball Tree 仅作为上述算法背景。

```text
KNN/
├── __init__.py   # 导出 KNeighborsClassifier 和 KDTree
├── knn.py        # 分类器、分块搜索、投票、概率计算
├── kdtree.py     # 距离、最大堆、同步排序、树的构建与查询
├── main.py       # Iris 分类示例
├── test_knn.py   # 已有回归测试
└── readme.md     # 本文
```

整体调用关系：

```text
fit(X, y)
  ├─ 参数与数据校验
  ├─ _resolve_metric：统一距离定义
  ├─ _check_algorithm：确定搜索后端
  ├─ np.unique：编码类别
  └─ 保存训练数据 / 建立 KDTree

kneighbors(X)
  ├─ brute   → argkmin → _argkmin_chunks
  └─ kd_tree → KDTree.query → 深度优先搜索 + 最大堆

predict(X) / predict_proba(X)
  ├─ brute 且 X 非 None → _brute_predict_proba
  │                       → argkmin_class_mode → 分块搜索后立即累计票数
  └─ 其他路径 → kneighbors → 权重 / 众数 / 归一化
```

## 3. 暴力搜索如何优化

### 3.1 使用约化距离，延迟开方

找最近邻只要求大小关系正确。由于正数上的开方和正次幂是单调的，可以使用约化距离：

$$
R_p(x,z)=\sum_j|x_j-z_j|^p.
$$

欧氏距离用平方距离排序；一般 Minkowski 距离用 $p$ 次幂之和排序；Manhattan 和 Chebyshev 无须额外转换。只有最终确实需要真实距离时，才对选出的 $k$ 个结果开方或取 $1/p$ 次幂。

这个优化同时用于暴力搜索和 KD-Tree。它减少距离变换次数，但不改变近邻排序。距离加权必须使用真实距离，否则会把 $1/D$ 错写为 $1/D^p$，改变投票含义。

### 3.2 欧氏距离转为矩阵运算

利用恒等式：

$$
\|x-y\|^2=\|x\|^2+\|y\|^2-2x^Ty,
$$

代码先计算每行的平方范数，再通过 `X @ Y.T` 得到所有点积，组合成平方距离矩阵。这样无须生成形状为 `(查询数, 训练数, 特征数)` 的差值张量。

这里关注的是代数变换与临时数组规模的降低，不讨论矩阵库的底层语言实现。浮点误差可能让理论上的零变成微小负数，因此用 `np.maximum(D, 0, out=D)` 截断到非负值。

其他距离先通过广播生成差值，原地取绝对值，再沿特征维求和、求最大值或求幂后求和。

### 3.3 只选 top-k，不对全部候选排序

`np.argpartition` 把最小的 $k$ 项放到前部，但不保证这 $k$ 项有序；随后只对它们调用 `np.argsort`。

一次选择的算法思路由全排序的 $O(n\log n)$，变成线性量级的选择加 $O(k\log k)$ 的局部排序。在 $k\ll n$ 时，可以省去大量无用的次序计算。

注意：距离并列时，近邻集合边界上的选择与“类别票数平票”是两个不同问题。这里没有额外规定所有等距邻居必须按训练索引排序。

### 3.4 查询轴与训练轴同时分块

只切分查询数据仍可能得到很宽的距离矩阵，因此 `_argkmin_chunks` 同时切分两条轴：

```text
对每个查询块 Xc：
    best_dist = k 个正无穷
    best_ind  = k 个占位索引
    对每个训练块 Yc：
        计算 Xc 与 Yc 的约化距离
        将当前 best 与新块候选拼接
        用 argpartition 保留最小的 k 项
    对最终 k 项排序
    yield 当前查询块的结果
```

正确性来自一个简单事实：已经被前面候选的 top-k 淘汰的元素，前面至少有 $k$ 个不比它远的元素；加入更多候选后，它也无须重新进入候选池。等距边界上可保留任意满足距离条件的 $k$ 项。

`_chunk_sizes` 按近似内存预算选择块大小。`working_memory` 在这里的单位是**字节**，底层函数默认 `1e6`：

- 欧氏距离按每对样本约 3 个 `float64` 数值估算。
- 其他距离按每对样本约 `max(d + 2, 3)` 个数值估算。
- 用预算估计可容纳的样本对数，再选择查询块大小和训练块大小。

预算控制的是分块尺度，不是进程内存的严格上限。候选索引、top-k 缓冲、数组拼接、最终输出和训练数据都还需要空间；块至少包含一个样本，极小预算也不意味着所有分配都能被压到预算以内。

设查询块大小为 $q$、训练块大小为 $b$，欧氏距离块为 $O(qb)$，一般距离差值张量为 $O(qbd)$，top-k 状态为 $O(qk)$。相比保存完整的距离矩阵，峰值临时空间可控得多。

### 3.5 搜索与分类累计衔接

单纯查询需要返回 `(m, k)` 的索引和可选距离。但分类最终只需要各类别票数。

`argkmin_class_mode` 在每个查询块完成近邻选择后，立即读取标签、计算权重、累计类别分数并归一化。使用 `np.add.at`，可以正确累计同一行中多个邻居投向同一类别的重复索引。

这条路径仍然保留当前查询块的 top-k，也仍会对该块结果排序；优化点是无需为整个查询集保存完整近邻结果后再做分类。最终概率矩阵仍占 $O(mC)$ 空间，其中 $C$ 为类别数。

## 4. KD-Tree 如何优化

### 4.1 数组化存储与平衡划分

树节点放在数组中，节点 `i` 的左右孩子为 `2*i+1` 和 `2*i+2`。训练数据保存在 `data`，树划分主要重排 `idx_array`，各节点用索引区间引用样本。

| 属性 | 含义 |
| --- | --- |
| `data` | 连续的 `float64` 训练数据 |
| `idx_array` | 树划分后的原始样本索引排列 |
| `idx_start`、`idx_end` | 每个节点对应的左闭右开索引区间 |
| `node_bounds` | 形状 `(2, 节点数, 特征数)`，存储包围盒下界和上界 |
| `is_leaf` | 是否为叶节点 |
| `radius` | 包围盒半边长向量的 p 范数；当前查询使用包围盒距离下界剪枝 |
| `n_levels`、`n_nodes` | 预先确定的树层数和节点数 |

代码根据样本数与 `leaf_size` 估计层数：

$$
L=\left\lfloor\log_2\left(\max\left(1,\frac{n-1}{\text{leaf\_size}}\right)\right)+1\right\rfloor,
\qquad N_{nodes}=2^L-1.
$$

每次选择样本跨度最大的特征作为划分维度，在该维的中位位置划分，使两个子节点的样本数接近。`leaf_size` 控制树深与叶内扫描量的折中，并不表示每片叶子都恰好具有这么多样本。

### 4.2 用选择代替建树时的完整排序

建树只需要让中位位置两边满足大小关系，无须把整个节点内部完全排序。`_partition_node_indices` 使用 quickselect 式划分，平均线性时间找到划分位置。

比较键是 `(指定维度上的坐标, 原始索引)`：坐标相同时按索引决定次序，使划分有确定的比较规则；这不等同于对整个区间做稳定排序。函数中的 `split_index` 是节点局部区间内的位置。

quickselect 的平均开销是 $O(s)$，其中 $s$ 是当前节点样本数；最坏情况不能直接保证线性。平衡树每层还要计算各节点的边界与特征跨度，典型建树成本可按 $O(ndL)$ 理解。

### 4.3 包围盒给出距离下界

设节点包围盒第 $j$ 维的区间是 $[l_j,u_j]$，查询点到这一维区间的最短距离为：

$$
g_j=\max(l_j-x_j,\;x_j-u_j,\;0).
$$

点在区间内时，该维贡献为 0；在区间外时，贡献为到最近边界的距离。因此查询点到整个包围盒的约化距离为：

$$
LB_p(x)=\sum_jg_j^p,\qquad LB_\infty(x)=\max_jg_j.
$$

盒内任何训练点的距离都不可能小于这个下界。如果下界已经大于当前第 $k$ 近邻的约化距离，那么整个节点都可以跳过。这是安全的分支限界搜索，不是近似近邻搜索。

### 4.4 固定大小最大堆

每个查询点维护容量为 $k$ 的最大堆：

- 堆保存目前最好的 $k$ 个候选。
- 堆顶是这些候选中距离最大的一个，即当前最差候选。
- 不优于堆顶的新点直接丢弃；更近的新点替换堆顶并向下调整。
- 查询最差距离为 $O(1)$，一次有效替换为 $O(\log k)$。

初始堆距离全为正无穷，让最早到来的真实候选自然进入堆。距离与原始索引始终同步移动，最终可以找回训练标签。

### 4.5 先搜索更近的孩子

`_query_single_depthfirst` 对当前节点依次处理：

1. 若节点距离下界大于堆顶，剪枝。
2. 若为叶节点，扫描节点内样本并更新堆。
3. 若为内部节点，分别计算两个孩子的下界，先访问下界较小的孩子，再访问另一个孩子。

先搜索更近的区域，可以更早找到好的候选，降低堆顶距离，从而加强后续剪枝。访问第二个孩子时，剪枝判断会使用更新后的堆。

在低维且数据分布合适时，树能减少大量点对距离计算；高维下包围盒区分能力下降，搜索可能接近全量扫描。KD-Tree 并不保证每次查询都是 $O(\log n)$。

### 4.6 最终结果的同步 introsort

最大堆只保证堆顶最大，并不保证所有邻居已按距离升序排列。输出前需要将距离和索引一起排序：

- 大区间采用三数取中、双向分区的快速排序。
- 小区间（长度不大于 15）采用插入排序。
- 初始深度预算为 `2 * floor(log2(k))`，预算耗尽时切换到堆排序。

这是 introsort 的思路：通常利用快速排序的效率，利用堆排序限制最坏时间复杂度，再利用插入排序处理小区间。这里只排序最终 $k$ 个邻居，排序时间为 $O(k\log k)$ 量级。

## 5. 自动选择、投票与边界语义

### 5.1 自动选择搜索后端

`_check_algorithm` 对 `algorithm='auto'` 使用以下规则，任一条件成立则选暴力搜索：

- 特征数 `n_features > 15`。
- `n_neighbors >= n_samples // 2`。
- 有效距离是 Minkowski，且 `p < 1`。

其余支持的距离使用 KD-Tree。这是启发式选择，不是在训练时运行性能基准。较大的 $k$ 会减弱树的剪枝效果，高维则会降低空间划分的收益。

显式指定 `brute` 或 `kd_tree` 可以固定路径。当前 `ball_tree` 参数会明确报出未实现提示。

### 5.2 距离参数归一化

`_resolve_metric` 将 Minkowski 的特殊情况转换为专门路径：`p=1` 转 Manhattan，`p=2` 转 Euclidean，`p=inf` 转 Chebyshev。`metric_params['p']` 优先于构造参数 `p`；显式命名的距离具有固定的范数，不受单独 `p` 参数改变。

这样后续计算无须反复判断等价距离，并能直接使用欧氏距离的矩阵公式。结果存入 `effective_metric_`、`effective_p_` 和 `effective_metric_params_`。

### 5.3 标签编码与平票

`fit` 使用 `np.unique(y, return_inverse=True)`，将原始标签转换为从 0 开始的连续类别编号。`classes_` 保存编号到原标签的映射，`_y` 保存每个训练样本的编号。

整数编号可以直接用作概率数组的列索引。输出概率第 `j` 列对应 `classes_[j]`，不能把列号直接理解为用户的原始标签。

`weighted_mode` 按排序后的候选类别遍历，仅在新类别票数**严格大于**已有最大票数时替换结果；`np.argmax` 同样选择首个最大位置。因此给定邻居集合时，类别票数相同会选择 `classes_` 中位置靠前的类别。

### 5.4 训练集自身查询

`kneighbors()` 或 `kneighbors(X=None)` 表示查询每个训练样本的其他邻居：内部先多取一个，再按原始索引排除样本自身。因此此时要求 `k + 1 <= n_samples_fit_`。

若重复样本导致返回的候选列表中没有当前样本自己的索引，则移除第一个候选，使返回数量仍然是 $k$。这里排除的是样本身份，不是删除所有零距离邻居。

显式调用 `kneighbors(X_train)` 则是普通外部查询，不自动排除自身。`predict(None)` 与 `predict_proba(None)` 也会进入自身排除的近邻路径。

## 6. knn.py：每个函数的作用

### 6.1 投票辅助函数

| 函数 | 作用、输入与输出 |
| --- | --- |
| `_get_weights(dist, weights)` | 将形状 `(m, k)` 的真实距离转换为权重。`None` 和 `uniform` 返回 `None` 表示等权；`distance` 计算倒数并处理零距离；可调用对象直接接收距离矩阵。 |
| `weighted_mode(a, w, axis=0)` | 沿指定轴累计每种取值的权重，返回加权众数及其总权重；保留被归约轴，长度为 1。严格大于的更新条件实现类别平票规则。 |
| `_mode(a, axis=0)` | 用全 1 权重调用 `weighted_mode`，得到普通众数及出现次数。 |

### 6.2 暴力搜索函数

| 函数 | 作用、输入与输出 |
| --- | --- |
| `_pairwise_reduced_distances(X, Y, metric, p)` | 计算两组样本间的约化距离矩阵，形状 `(len(X), len(Y))`。欧氏距离走范数与点积公式，其他距离走广播差值与归约。 |
| `_rdist_to_dist(distances, metric, p)` | 将欧氏平方距离开方、一般 Minkowski 约化距离取 `1/p` 次幂，其余距离原样返回。 |
| `_chunk_sizes(n_queries, n_fit, n_features, metric, working_memory)` | 根据近似字节预算估计查询块和训练块大小；拒绝非正预算。 |
| `_argkmin_chunks(X, Y, k, metric, p, working_memory)` | 双轴分块搜索的生成器，逐块产出 `(start, stop, reduced_dist, indices)`；距离和索引均为当前查询块的已排序 top-k。 |
| `argkmin(X, Y, k, metric, p, return_distance=True, working_memory=1e6)` | 暴力近邻查询入口；转换数组、校验维度和 k，收集各块结果。返回真实距离与索引，或只返回索引。 |
| `argkmin_class_mode(X, Y, y, k, n_classes, metric, p, weights, working_memory=1e6)` | 分块搜索后立即读取编码标签并累计权重，返回 `(m, n_classes)` 的概率矩阵；拒绝某个查询的邻居权重全部为零的情况。名称虽有 mode，返回值实际是概率。 |

### 6.3 KNeighborsClassifier 的所有方法

| 方法 | 作用 |
| --- | --- |
| `__init__(...)` | 保存邻居数、权重、算法、叶大小、距离及相关参数，尚不训练。 |
| `_resolve_metric()` | 检查距离名称与参数，处理 Minkowski 特例，记录有效距离属性。 |
| `_check_algorithm(X)` | 根据显式配置或自动规则决定 `_fit_method`，并校验树与距离的组合。 |
| `fit(X, y)` | 校验样本、标签、参数与有限数值；编码类别；记录样本数和特征数；保存数据或建树；返回 `self`。 |
| `kneighbors(X=None, n_neighbors=None, return_distance=True)` | 公共近邻查询入口；允许临时指定 k，分派搜索后端，处理 `X=None` 的自身排除；返回形状 `(m, k)` 的结果。一维外部查询会转换为一行。 |
| `_check_is_fitted()` | 检查是否已有 `_fit_method`；未拟合时给出错误。 |
| `_brute_predict_proba(X)` | 校验并转换外部查询，调用分块搜索与投票相衔接的 `argkmin_class_mode`。 |
| `predict(X)` | 暴力外部查询先计算概率并取最大列；其他路径查近邻、求普通或加权众数；最终映射回原始标签并返回一维数组。 |
| `predict_proba(X)` | 返回按 `classes_` 排列的类别概率。暴力外部查询走分块累计；其他路径逐邻居将权重加入类别列，再按行归一化。 |
| `score(X, y)` | 计算预测标签与给定标签相等的比例，返回浮点准确率。 |

主要构造参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `n_neighbors` | `5` | 查询与投票使用的邻居数量 |
| `weights` | `'uniform'` | 等权、反距离权重，或接收距离矩阵的函数；`None` 也是等权语义 |
| `algorithm` | `'auto'` | 自动选择，或显式指定 `'brute'`、`'kd_tree'` |
| `leaf_size` | `30` | 分类器创建 KD-Tree 时使用的叶规模参数 |
| `p` | `2` | Minkowski 指数 |
| `metric` | `'minkowski'` | 距离类型 |
| `metric_params` | `None` | 距离附加参数；这里支持 Minkowski 的 `p` 覆盖 |
| `n_jobs` | `None` | 接口中保存该参数；当前代码没有据此调度并行任务 |

本目录的输入路径面向稠密数值特征和一维单输出分类标签。自定义权重函数应返回与距离矩阵相容的有效权重，通常应使用有限、非负值，并保证每行总权重为正。代码对全零邻居权重行有显式检查。

## 7. kdtree.py：每个函数的作用

### 7.1 距离与节点下界

| 函数 | 作用 |
| --- | --- |
| `rdist(x1, x2, p)` | 计算两个向量的约化距离；有限 p 返回绝对差的 p 次幂之和，无穷 p 返回最大绝对差。 |
| `dist(x1, x2, p)` | 在 `rdist` 基础上恢复真实距离，提供单点距离接口。 |
| `rdist_to_dist(r, p)` | 将标量或数组形式的约化距离转换为真实距离，供查询输出使用。 |
| `min_rdist(node_bounds, i_node, pt, p)` | 计算点到节点包围盒的约化距离下界，为搜索剪枝和孩子访问顺序提供依据。 |

### 7.2 最大堆

| 函数 / 方法 | 作用 |
| --- | --- |
| `heap_push(values, indices, val, val_idx)` | 尝试把一个候选插入固定大小最大堆；若 `val >= 堆顶` 则丢弃，否则替换根并向下调整，同步更新索引。 |
| `NeighborsHeap.__init__(n_pts, n_nbrs)` | 为每个查询分配一行大小为 k 的堆，距离初始化为正无穷，索引初始化为 0。 |
| `NeighborsHeap.largest(row)` | 返回该查询当前最大候选距离，即堆顶剪枝阈值。 |
| `NeighborsHeap.push(row, val, i_val)` | 为指定查询行调用 `heap_push`。 |
| `NeighborsHeap.get_arrays(sort=True)` | 按需逐行同步排序，然后返回内部距离数组和索引数组。排序发生在搜索结束后，此时不再需要最大堆结构。 |

### 7.3 同步排序

以下排序区间均采用左闭右开的 `[lo, hi)`，距离与索引始终保持配对。

| 函数 | 作用 |
| --- | --- |
| `_swap(values, indices, i, j)` | 同时交换两个位置的距离和索引。 |
| `_simultaneous_sort(values, indices)` | 同步排序入口；处理空数组，设置 introsort 深度预算，对整行排序。 |
| `_insertion_sort(values, indices, lo, hi)` | 对小区间执行插入排序，通过移动元素完成局部升序排列。 |
| `_inplace_median3(values, indices, lo, hi)` | 对首、中、末三个候选进行比较交换，把中值枢轴放在分区需要的位置，返回枢轴值。 |
| `_sift_down(values, indices, lo, root, end)` | 在偏移为 lo 的子区间中恢复最大堆；root 和 end 使用相对于 lo 的位置，支持非零起点的堆排序。 |
| `_heapsort(values, indices, lo, hi)` | 对区间建最大堆，再反复将最大值放到末端，得到升序结果，作为 introsort 的后备算法。 |
| `_introsort_2way(values, indices, lo, hi, maxd)` | 综合快速排序、深度预算、堆排序回退和小区间插入排序；左区间递归处理，右区间通过循环继续处理。 |

### 7.4 建树划分

| 函数 | 作用 |
| --- | --- |
| `_partition_node_indices(data, idx_array, start, end, split_dim, split_index)` | 对当前节点的索引区间做 quickselect 式划分，使局部中位位置两边满足比较关系，再写回全局索引数组。 |
| 上述函数内部的 `less(a, b)` | 比较样本 a、b 在 split_dim 维的坐标；相同坐标用原始索引打破并列，提供统一比较规则。 |

### 7.5 KDTree 的所有方法

| 方法 | 作用 |
| --- | --- |
| `__init__(data, leaf_size=40, metric='minkowski', p=2)` | 校验并保存数据，解析距离，估计树规模，分配节点数组并启动建树。直接创建树的默认叶大小为 40，分类器则显式传入自身默认值 30。 |
| `_init_node(i_node, idx_start, idx_end)` | 计算该节点样本的逐维上下界、包围盒半径，并记录索引范围。 |
| `_find_node_split_dim(idx_start, idx_end)` | 计算各维最大值减最小值，选择跨度最大的维度；并列时使用 `argmax` 的首个位置。 |
| `_recursive_build(i_node, idx_start, idx_end)` | 初始化节点；达到预分配树底层或样本不足以再分时标记叶子，否则按中位位置划分并递归构建两个孩子。 |
| `_rdist(i_point, pt)` | 从训练数据取出原始索引对应的点，调用 `rdist` 计算它与查询点的约化距离。 |
| `_query_single_depthfirst(i_node, pt, i_pt, heap, reduced_dist_lb)` | 对一个查询执行深度优先分支限界；根据下界剪枝，在叶子更新候选堆，内部节点先访问下界较小的孩子。 |
| `query(X, k=1, return_distance=True)` | 查询入口；校验输入与 k，把最后一维视为特征维，展平前面的查询维度，逐点搜索、排序，并恢复 `X.shape[:-1] + (k,)` 的输出形状。只要索引时跳过真实距离转换。 |

`VALID_METRICS` 声明四种支持的距离名称；`knn.py` 中的 `KD_VALID_METRICS` 将其转换为集合，供参数校验使用。

## 8. 示例、包入口与测试函数

`__init__.py` 没有自定义函数，通过 `__all__` 暴露 `KNeighborsClassifier` 和 `KDTree`，因此可以直接从 `KNN` 导入这两个类。

### 8.1 main.py

`main()` 完成以下流程：加载 Iris 数据，将 30% 样本划作测试集，在训练集拟合标准化器，以 `k=3` 训练分类器，输出搜索后端、准确率和分类报告。文件末尾的主模块判断负责启动此函数。

现有示例的最后一段还会创建 sklearn 分类器并打印结果比较；这是示例本身的行为。本文仅说明其职责，不执行该比较，也不据此作源码一致性结论。

### 8.2 test_knn.py

现有 `KNNParityTests` 的函数职责如下。此处是测试内容说明，并非本次运行结果。

| 方法 | 覆盖内容 |
| --- | --- |
| `setUpClass()` | 使用固定随机种子准备 80 个四维训练样本、四类标签和 19 个查询样本，供多个测试复用。 |
| `assert_matches_sklearn(**params)` | 公共断言辅助方法，构建两套分类器并比较距离、邻居索引、标签和概率。 |
| `test_main_iris_path()` | 覆盖 Iris 划分、标准化、自动选择 KD-Tree 以及分类输出。 |
| `test_named_metrics_ignore_standalone_p()` | 覆盖显式命名距离不受独立 p 参数影响，以及不同后端、权重组合。 |
| `test_minkowski_metrics_and_fused_brute_vote()` | 覆盖 p 为 1、2、3、无穷时的不同算法和权重路径，包括暴力分类累计。 |
| `test_brute_force_two_dimensional_chunking()` | 用 128 字节的极小近似预算触发双轴分块，覆盖多种距离的近邻查询。 |
| `test_auto_uses_brute_for_semimetric_and_ball_tree_is_explicit()` | 覆盖 p 小于 1 时自动选择暴力搜索，以及显式 Ball Tree 请求的错误提示。 |
| `test_kdtree_named_metric_and_introsort_fallback()` | 覆盖 KD-Tree 命名距离，以及非零起点区间的堆排序回退；检查区间外数据不变、区间内有序、距离与索引配对正确。 |

## 9. 使用方式

在项目根目录运行。核心实现依赖 NumPy；已有 Iris 示例和测试另外使用 scikit-learn。

下面的最小示例仅使用本目录分类器：

```python
import numpy as np
from KNN import KNeighborsClassifier

X = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
y = np.array(["A", "A", "B", "B"])
query = np.array([[0.1, 0.2], [2.1, 2.2]])

model = KNeighborsClassifier(
    n_neighbors=2,
    algorithm="kd_tree",
    weights="distance",
)
model.fit(X, y)

print(model.predict(query))        # ['A' 'B']
print(model.classes_)              # ['A' 'B']，同时也是概率列的顺序
print(model.predict_proba(query))
distances, indices = model.kneighbors(query)
print(distances)
print(indices)                    # X 中的原始行索引
```

已有 Iris 示例可用 `python -m KNN.main` 或 `python KNN/main.py` 运行；已有测试入口为 `python -m unittest KNN.test_knn -v`。目录名区分大小写，模块路径使用 `KNN`。这些命令仅供读者使用，本次文档编写未执行比较测试。

## 10. 优化与成本对照

以下以 $n$ 表示训练样本数、$m$ 表示查询数、$d$ 表示特征数、$C$ 表示类别数、$q$ 和 $b$ 表示查询块与训练块大小。

| 环节 | 优化 | 主要收益与成本 |
| --- | --- | --- |
| 距离计算 | 约化距离 | 排序和剪枝阶段省去开方；需要真实距离时只转换最终邻居 |
| 欧氏距离 | 范数与点积展开 | 避免 `(q,b,d)` 差值张量，距离块为 `(q,b)` |
| 暴力候选选择 | `argpartition` 加 top-k 局部排序 | 不对全部训练距离建立完整次序 |
| 暴力搜索内存 | 查询轴、训练轴双重分块 | 控制临时数组规模；距离计算总量仍为 $O(mnd)$ |
| 分块 top-k 合并 | 只保留历史最优 k 项 | 额外选择工作约为 $O(m(n+\lceil n/b\rceil k))$，最后排序约为 $O(mk\log k)$；块过小会增加合并次数 |
| 分类输出 | 查询块完成后立即累计类别权重 | 无须长期保存整个查询集的近邻结果；仍要保存 $O(mC)$ 概率输出 |
| KD-Tree 构建 | 最大跨度维度与中位划分 | 树形平衡，选择过程避免每个节点完整排序 |
| KD-Tree 查询 | 包围盒下界、近孩子优先 | 在适合的数据上减少扫描；最坏仍可能遍历全部样本 |
| 候选维护 | 容量 k 的最大堆 | $O(1)$ 获取剪枝阈值，有效更新为 $O(\log k)$ |
| 查询结果排序 | 同步 introsort | 只排序 k 项，并保持距离与索引配对 |
| 分类标签 | 连续整数编码 | 便于数组累计、概率列定位与原标签恢复 |
| 后端决策 | `auto` 启发式 | 根据维度、k 和距离条件选择合适搜索路径 |

若一次 KD-Tree 查询访问 $u$ 个节点、扫描 $s$ 个样本、发生 $h$ 次有效堆更新，其成本可理解为 $O(ud+sd+h\log k+k\log k)$；树的优势主要来自让 $s$ 明显小于 $n$，而不是改变分类投票公式。

阅读代码时，可以先沿 `fit → kneighbors → predict` 理解职责，再读 `_argkmin_chunks` 理解分块 top-k，最后沿 `KDTree.query → _query_single_depthfirst → heap_push` 理解空间剪枝与候选维护。
