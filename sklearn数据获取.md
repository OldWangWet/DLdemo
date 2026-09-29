# sklearn 数据获取与使用

## 1. 本环境的检查结果

- 检查日期：2026-09-29。
- Conda 环境：`DLdemo`。
- 环境路径：`/home/fire/miniconda3/envs/DLdemo`。
- sklearn 版本：`1.9.1`。
- 数据集模块：`/home/fire/miniconda3/envs/DLdemo/lib/python3.13/site-packages/sklearn/datasets/__init__.py`。

本文依据本环境中 `sklearn.datasets` 的实际导出接口和函数文档整理。下表中的六个内置数值数据集已实际加载并确认形状；远程数据集仅检查了接口，未下载验证。

项目中执行 Python 命令前，应在同一次 Bash 调用中初始化并激活环境：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python your_script.py
```

## 2. 随库附带的数据集：load_*

以下六个数值数据集无需联网下载：

| 加载函数 | 数据集 | 样本数 | 特征数 | 任务 |
|---|---|---:|---:|---|
| `load_iris()` | 鸢尾花 | 150 | 4 | 三分类 |
| `load_digits()` | 手写数字，8×8 灰度图 | 1,797 | 64 | 十分类，标签为 0～9 |
| `load_wine()` | 葡萄酒化学成分 | 178 | 13 | 三分类 |
| `load_breast_cancer()` | 乳腺癌诊断 | 569 | 30 | 二分类：恶性 / 良性 |
| `load_diabetes()` | 糖尿病 | 442 | 10 | 回归，预测疾病进展指标 |
| `load_linnerud()` | 体能锻炼与生理指标 | 20 | 3 | 多目标回归，包含 3 个预测目标 |

### 常用加载方式

```python
from sklearn.datasets import load_wine

# 直接获取特征矩阵 X 和标签 y
X, y = load_wine(return_X_y=True)
print(X.shape)  # (178, 13)
print(y.shape)  # (178,)

# 获取包含数据及说明的 Bunch 对象，可通过属性访问
data = load_wine()
print(data.data.shape)
print(data.target.shape)
print(data.feature_names)  # 特征名称
print(data.target_names)   # 类别名称；并非所有数据集都有此属性
print(data.DESCR)          # 数据集说明
```

这里的 `X` 通常具有形状 `(样本数, 特征数)`。单目标任务的 `y` 通常为 `(样本数,)`；`load_linnerud()` 的 `y` 为 `(20, 3)`，三个目标分别是体重、腰围和脉搏。

### 示例照片

还提供两张用于图像处理的示例照片，它们不是带分类标签的训练数据集：

```python
from sklearn.datasets import load_sample_image, load_sample_images

china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
images = load_sample_images()  # 同时加载两张图片
```

### 并非所有 load_* 都代表内置数据集

以下函数读取用户提供的文件，本身不附带某个固定数据集：

- `load_files()`：读取按类别子目录组织的文本文件。
- `load_svmlight_file()`：读取 SVMlight / LIBSVM 格式的数据文件。
- `load_svmlight_files()`：读取多个上述格式的数据文件。

## 3. 通常需要下载的数据集：fetch_*

以下接口在本环境中可用，但对应数据本身不随 sklearn 一起安装。首次使用通常需要联网，已有本地缓存时通常可以复用。

| 函数 | 数据内容及用途 |
|---|---|
| `fetch_california_housing()` | 加州房价，回归 |
| `fetch_20newsgroups()` | 20 类新闻组文本，文本分类 |
| `fetch_20newsgroups_vectorized()` | 已向量化的新闻组文本 |
| `fetch_olivetti_faces()` | Olivetti 人脸图像 |
| `fetch_lfw_people()` | LFW 人脸图像，人物分类 |
| `fetch_lfw_pairs()` | LFW 人脸配对，判断是否为同一人 |
| `fetch_covtype()` | 森林覆盖类型分类 |
| `fetch_kddcup99()` | 网络入侵检测 |
| `fetch_rcv1()` | 新闻文本，多标签分类 |
| `fetch_species_distributions()` | 物种地理分布 |
| `fetch_openml()` | 从 OpenML 获取数据的通用接口，并非单一数据集 |

例如，下载并加载加州房价数据：

```python
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
```

本环境还导出了 `fetch_file()`，它用于下载并缓存文件，不代表一个固定的数据集。

## 4. 按需生成模拟数据：make_*

这类函数通过参数生成数据，不是读取已有的数据集，适合算法演示、可视化和可控实验。

| 用途 | 本环境提供的函数 |
|---|---|
| 分类 | `make_classification()`、`make_multilabel_classification()`、`make_gaussian_quantiles()`、`make_hastie_10_2()` |
| 回归 | `make_regression()`、`make_friedman1()`、`make_friedman2()`、`make_friedman3()`、`make_sparse_uncorrelated()` |
| 点簇及二维形状 | `make_blobs()`、`make_moons()`、`make_circles()` |
| 降维与流形学习 | `make_s_curve()`、`make_swiss_roll()` |
| 双聚类结构 | `make_biclusters()`、`make_checkerboard()` |
| 矩阵及信号 | `make_low_rank_matrix()`、`make_sparse_coded_signal()`、`make_sparse_spd_matrix()`、`make_spd_matrix()` |

例如：

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)
```

