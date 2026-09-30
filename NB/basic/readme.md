# 基础手动实现

`nb.py` 使用 NumPy 按 [上层 README](../readme.md) 的公式实现五种朴素贝叶斯，提供 `fit(X, y)` 和 `predict(X)`。模型计算本身不调用 sklearn；演示和测试使用 sklearn 加载数据、划分数据、计算准确率并对比结果。

从项目根目录运行：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python NB/basic/main.py
```

也可以将最后的命令替换为 `python -m NB.basic.main`。

## 与数据集的对应关系

使用与 `NB/main.py` 相同的手写数字数据、分层划分、30% 测试集和随机种子 42。

| 模型 | 特征表示与计算 |
|---|---|
| `GaussianNB` | 原始灰度数值；按类别计算均值和最大似然方差 |
| `MultinomialNB` | 非负灰度作为计数式权重；按类别累计并平滑 |
| `CategoricalNB` | 每个像素的整数灰度作为类别编码；`min_categories=17` 保留全部等级 |
| `BernoulliNB` | 灰度大于 8 记为 1，否则为 0；同时计算出现和未出现的概率 |
| `ComplementNB` | 累计其他类别的灰度权重；比较补集概率的负对数得分，不加类别先验 |

灰度值不是实际事件计数，也不保证满足高斯假设，因此这些结果用于算法演示。

## 实现范围

- 使用对数求和，避免直接连乘概率导致下溢。
- 高斯模型默认 `var_smoothing=1e-9`：将训练集各特征方差的最大值乘以该参数，加到每类每个特征的方差上，处理恒为零的像素。最终方差仍为零时会报错。
- 离散模型默认 `alpha=1.0`，本阶段要求该参数严格大于 0。
- 类别模型要求非负整数编码；`min_categories` 支持一个正整数，预测编码需在拟合确定的范围内。
- 伯努利模型支持固定阈值，或以 `binarize=None` 接收已二值化数据。
- 补集模型仅实现 README 的 `norm=False` 方式，要求至少两个类别。
- 本阶段支持稠密二维特征矩阵和一维标签，未实现稀疏输入、样本权重、增量训练、概率输出及完整 sklearn 接口；后续两个阶段继续扩展。

## 验证

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m unittest NB.basic.test_nb
```

测试覆盖 README 示例、手写数字上与 sklearn 的预测及参数对比，以及预留类别和二值化阈值。
