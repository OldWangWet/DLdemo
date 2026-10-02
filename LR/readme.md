# 线性回归：从贝叶斯公式到 MSE 与正则化

## 一、贝叶斯公式：似然与先验共同决定参数

设有 $m$ 个样本、每个样本有 $n$ 个特征，数据集为 $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{m}$，权重为 $w\in\mathbb{R}^n$。将输入视为已知，贝叶斯公式为：

$$
p(w\mid\mathcal{D})
=\frac{p(\mathcal{D}\mid w)\,p(w)}{p(\mathcal{D})}
$$

- **先验 $p(w)$：** 观察数据前，对参数的判断。
- **似然 $p(\mathcal{D}\mid w)$：** 给定参数时，观测数据的概率密度。
- **后验 $p(w\mid\mathcal{D})$：** 结合数据后，对参数的判断。
- **证据 $p(\mathcal{D})$：** 归一化因子，与待优化参数 $w$ 无关。

$$
p(\mathcal{D})=\int p(\mathcal{D}\mid w)p(w)\,dw
$$

最大后验估计（MAP）选择后验密度最大的参数。去掉分母，得到：

$$
\hat{w}_{\mathrm{MAP}}
=\arg\max_w p(w\mid\mathcal{D})
=\arg\max_w p(\mathcal{D}\mid w)\,p(w)
$$

因为对数单调，最大化乘积等价于最小化负对数之和：

$$
\hat{w}_{\mathrm{MAP}}
=\arg\min_w
\left[-\log p(\mathcal{D}\mid w)-\log p(w)\right]
$$

**最大化「似然 × 先验」，等价于最小化「负对数似然 + 负对数先验」。** 接下来分别说明两项如何成为数据损失与正则项。

## 二、两个模型假设

**假设一：线性关系。** 目标的条件均值是特征的线性函数：

$$
\mathbb{E}[y_i\mid x_i]=w^\top x_i
$$

这里的“线性”指对参数线性，不要求输入特征服从某种分布。

**假设二：高斯噪声。** 给定输入，各样本噪声相互独立，且具有相同方差。两项假设合起来得到：

$$
y_i = w^\top x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

因此，预测值是 $\hat{y}_i=w^\top x_i$，观测值围绕预测值高斯波动：

$$
y_i\mid x_i,w\sim\mathcal{N}(w^\top x_i,\sigma^2)
$$

为简化推导，本文省略截距。实际模型可另加截距，通常不对截距正则化。以下把噪声方差 $\sigma^2$ 视为固定。

## 三、负对数似然 → MSE：如何求解参数

### 3.1 高斯噪声导出平方误差

单样本的条件概率密度为：

$$
p(y_i\mid x_i,w)
=\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\left(-\frac{(y_i-w^\top x_i)^2}{2\sigma^2}\right)
$$

**噪声独立，所以联合似然是各样本密度的乘积：**

$$
p(\mathcal{D}\mid w)
=\prod_{i=1}^{m}p(y_i\mid x_i,w)
$$

取负对数，乘积变成求和：

$$
-\log p(\mathcal{D}\mid w)
=\frac{m}{2}\log(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i-w^\top x_i)^2
$$

均方误差（MSE）定义为：

$$
\mathrm{MSE}(w)
=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

因此：

$$
-\log p(\mathcal{D}\mid w)
=C+\frac{m}{2\sigma^2}\mathrm{MSE}(w)
$$

$C$ 与 $w$ 无关，比例系数为正，故**最小化负对数似然等价于最小化 MSE**。

推导中，**高斯分布产生平方项，独立性使各样本的损失相加，同方差使它们等权**。这解释了 MSE 的概率来源；也可以直接选择 MSE 作为拟合目标，而不先假设噪声分布。

### 3.2 最小二乘目标

计算 MSE：先求每个样本的残差 $y_i-\hat{y}_i$，再平方、求和、除以样本数。例如残差为 $1,-2,0$ 时，MSE 为 $(1+4+0)/3=5/3$。

训练则要找到使 MSE 最小的 $w$，这就是普通最小二乘（OLS）。为方便求导，常使用：

$$
J(w)=\frac{1}{2m}\sum_{i=1}^{m}(w^\top x_i-y_i)^2
=\frac{1}{2}\mathrm{MSE}(w)
$$

残差平方和（SSE）、MSE 和 $J$ 只差正的常数倍，最优参数相同。**最小二乘定义目标，正规方程与梯度下降负责求解。**

### 3.3 正规方程：直接求解

令矩阵 $X$ 的第 $i$ 行为 $x_i^\top$，向量 $y$ 收集所有目标值：

$$
J(w)=\frac{1}{2m}\|Xw-y\|_2^2,
\qquad
\nabla_wJ=\frac{1}{m}X^\top(Xw-y)
$$

令梯度为 0：

$$
X^\top X\hat{w}=X^\top y
$$

若 $X$ 的列线性无关，则：

$$
\hat{w}=(X^\top X)^{-1}X^\top y
$$

这是直接解，不需要学习率。若矩阵不可逆，求逆公式不适用，但最小二乘仍有解，参数可能不唯一。

### 3.4 梯度下降：迭代求解

从初始参数出发，沿梯度反方向更新：

$$
w^{(t+1)}=w^{(t)}-\eta_tg_t
$$

$\eta_t>0$ 是学习率，$g_t$ 是梯度或其估计。学习率过小收敛慢，过大可能震荡或发散。平方误差目标是凸函数，但收敛仍需要合适的学习率。

**BGD（批量梯度下降）：** 每次使用全部样本。

$$
g_t=\frac{1}{m}X^\top(Xw^{(t)}-y)
$$

**SGD（随机梯度下降）：** 每次使用一个样本 $i$。

$$
g_t=(x_i^\top w^{(t)}-y_i)x_i
$$

均匀随机抽样时，单样本梯度的期望等于完整梯度，但单步损失可能上升。固定学习率下，参数可能在最优点附近波动，因此常逐渐减小学习率。

**Mini-batch（小批量梯度下降）：** 每次使用批次 $B_t$，按实际样本数 $q$ 求平均。

$$
g_t=\frac{1}{q}\sum_{i\in B_t}(x_i^\top w^{(t)}-y_i)x_i
$$

$q=1$ 对应 SGD，$q=m$ 对应 BGD；最后一批也按实际样本数求平均。

| 方法 | 每次使用样本数 | 单次计算量 | 梯度波动 | 每轮更新次数 |
|---|---|---|---|---|
| BGD | $m$ | 大 | 无抽样波动 | 1 |
| SGD | 1 | 小 | 较大 | $m$ |
| Mini-batch | 通常为 $q$ | 介于两者之间 | 通常较小 | $\lceil m/q\rceil$ |

一轮（epoch）指遍历训练集一次，一步（step）指更新一次参数。表中按逐轮遍历计算；SGD 和 mini-batch 通常每轮先打乱样本。

## 四、负对数先验 → 正则化

似然描述噪声，先验约束参数；这是两种不同的分布假设。下面保持高斯噪声不变，只改变权重先验。

### 4.1 高斯先验 → L2 正则（Ridge）

假设各权重独立，且服从零均值高斯分布：

$$
p(w_j)=\frac{1}{\sqrt{2\pi}\tau}
\exp\!\left(-\frac{w_j^2}{2\tau^2}\right),
\qquad
p(w)=\prod_{j=1}^{n}p(w_j)
$$

取负对数：

$$
-\log p(w)=\frac{1}{2\tau^2}\sum_{j=1}^{n}w_j^2+C
$$

代入第一节的 MAP 目标，忽略常数：

$$
\arg\min_w
\left[
\frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i-w^\top x_i)^2
+\frac{1}{2\tau^2}\sum_{j=1}^{n}w_j^2
\right]
$$

同乘 $2\sigma^2$，得到 Ridge 目标：

$$
\arg\min_w\left[\mathrm{SSE}(w)+\lambda\|w\|_2^2\right],
\qquad
\lambda=\frac{\sigma^2}{\tau^2}
$$

**作用：** 将系数向 0 收缩，缓解相关特征带来的估计不稳定，通常不产生稀疏解。噪声越大、先验方差越小，正则化越强。

### 4.2 拉普拉斯先验 → L1 正则（Lasso）

假设各权重独立，且服从零均值拉普拉斯分布，尺度为 $b>0$：

$$
p(w_j)=\frac{1}{2b}\exp\!\left(-\frac{|w_j|}{b}\right),
\qquad
p(w)=\prod_{j=1}^{n}p(w_j)
$$

取负对数：

$$
-\log p(w)=\frac{1}{b}\sum_{j=1}^{n}|w_j|+C
$$

代入 MAP 并同乘 $2\sigma^2$：

$$
\arg\min_w\left[\mathrm{SSE}(w)+\lambda\|w\|_1\right],
\qquad
\lambda=\frac{2\sigma^2}{b}
$$

**作用：** 收缩系数，并使部分系数精确为 0，从而选择特征。改变的是参数先验，数据损失仍然是平方误差。

### 4.3 为什么 L1 能得到精确的 0

L2 惩罚的导数随权重接近 0 而减小；L1 在非零处的斜率大小固定，在 0 处有尖点。

用一维问题对比：

$$
\min_w\left[\frac{1}{2}(y-w)^2+\lambda|w|\right]
\quad\Longrightarrow\quad
\hat{w}=\mathrm{sign}(y)\max(|y|-\lambda,0)
$$

当 $|y|\le\lambda$ 时，L1 的解为 0，这是软阈值机制。

$$
\min_w\left[\frac{1}{2}(y-w)^2+\frac{\lambda}{2}w^2\right]
\quad\Longrightarrow\quad
\hat{w}=\frac{y}{1+\lambda}
$$

在这个例子中，有限的 $\lambda$ 只把非零的 $y$ 缩小。一般多维问题中 L2 系数也可能为 0，但没有上述阈值机制。拉普拉斯先验是连续分布，稀疏性来自 MAP 最优点。

| 对照项 | L2：Ridge | L1：Lasso |
|---|---|---|
| 参数先验 | 高斯 | 拉普拉斯 |
| 惩罚项 | $\sum_jw_j^2$ | $\sum_j\vert w_j\vert$ |
| 主要效果 | 收缩、稳定系数 | 收缩、特征选择 |
| 在 0 处 | 可导 | 不可导 |

**系数约定：** 上述正则目标使用 SSE。若改成 MSE 或 $J$，需同时缩放惩罚项，才能保持同一目标。例如整个目标除以 $m$ 后，对应 $\mathrm{MSE}+\lambda\Omega(w)/m$，其中 $\Omega(w)$ 是 L1 或 L2 惩罚。

## 五、学习要点

- **标准化：** 特征尺度影响梯度下降和系数惩罚；标准化统计量只用训练集计算。
- **泛化：** 训练误差小不保证预测新数据时误差小。正则可缓解过拟合，过强则可能欠拟合；用验证集调参。
- **残差：** 残差的弯曲趋势、漏斗形、时间相关性，分别提示均值关系、同方差、独立性假设可能不合适。
- **MAP：** 得到一组最优参数；完整贝叶斯预测还会对参数的不确定性积分。

## 参考

- [线性模型的概念与目标函数](https://scikit-learn.org/stable/modules/linear_model.html)
- [随机梯度下降的数学说明](https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation)
