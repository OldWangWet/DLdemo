# sklearn 源码版朴素贝叶斯

本目录按照[项目 README](../../readme.md) 的第三阶段建设，直接参考 **scikit-learn 1.9.1** 的源码，不导入 `basic` 或 `optimized` 的模型。实现五种模型的全部构造参数、训练与预测计算、增量训练、样本权重、先验及对应稀疏路径，并手动实现目标模型使用的公共校验和 estimator 流程。

**当前状态：** NumPy/SciPy 主要模型路径已完成并通过逐项对照；支持 pandas 输入、常规 sklearn 组合器和简单消费者元数据请求。仍存在公共框架兼容差异及未验证后端，详见第 7 节，因此**不标记为 sklearn 1.9.1 全范围完整复现**。

模型及其依赖文件不导入 sklearn，训练和预测没有调用 sklearn 模型、校验工具、标签编码器或基类。sklearn 只出现在演示、测试和资源测量中，用作数据加载、组合器和对照。计算使用 Python、NumPy/SciPy；不增加源码没有的缓存、输入归一化、并行或资源策略。

## 1. 参考源码和目录

参考环境：Python 3.13.15、NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1。源码根目录为 `/home/fire/miniconda3/envs/DLdemo/lib/python3.13/site-packages/sklearn/`。

| 本目录文件/功能 | sklearn 1.9.1 源码依据 |
|---|---|
| `nb.py` 五种模型及公共预测 | `naive_bayes.py`：`_BaseNB`、`GaussianNB`、`_BaseDiscreteNB` 及四种离散类 |
| 参数、克隆、标签、拟合装饰器 | `base.py`：`BaseEstimator`、`ClassifierMixin`、`_fit_context`；`utils/_param_validation.py`、`utils/_tags.py` |
| 输入、特征名、权重、拟合状态 | `utils/validation.py`：`validate_data`、`check_array`、`_check_y`、`_check_sample_weight`、`check_is_fitted` |
| 类别契约和标签编码 | `utils/multiclass.py`：`unique_labels`、`_check_partial_fit_first_call`；`preprocessing/_label.py`：`LabelBinarizer`、`label_binarize` |
| 二值化、稀疏乘法 | `preprocessing/_data.py`：`binarize`；`utils/extmath.py`：`safe_sparse_dot` 的二维路径 |
| namespace、设备、均值和概率归一化 | `utils/_array_api.py`：`get_namespace_and_device`、`move_to`、`_average`、`_logsumexp` |
| 元数据消费者请求 | `utils/_metadata_requests.py`：`MethodMetadataRequest`、`MetadataRequest`、`RequestMethod` 的 setter 流程 |
| 本地配置 | `_config.py` 中目标模型相关的五个设置 |

改编代码保留 BSD-3-Clause 作者和许可信息，见 [LICENSE](LICENSE)。`nb.py` 保留源码计算顺序；公共流程按目标模型实际调用的路径实现，不复刻整个 sklearn 工具库。

```text
NB/sklearn_impl/
├── __init__.py
├── nb.py                     # 五种模型；不依赖前两阶段
├── _support.py               # 参数、校验、标签编码、矩阵/数组公共流程
├── _config.py                # 独立本地配置
├── _metadata.py              # 元数据简单消费者协议
├── main.py                   # digits：fit/partial_fit 与 sklearn 对照
├── test_nb.py                # 36 项测试，多组参数子测试
├── verify_upstream.py        # 原始 NB 测试及公共 estimator 检查
├── verification_results.json
├── benchmark.py              # 稀疏存储、分配峰值、分批状态验证
├── benchmark_results.json
├── requirements-test.txt     # 可选扩展验证依赖
├── LICENSE
└── readme.md
```

## 2. 功能与行为对照项

各类构造参数名、默认值和只允许关键字的调用方式与目标源码一致。

| 模型 | 参数 | 数据及训练能力 | 拟合属性 |
|---|---|---|---|
| `GaussianNB` | `priors`、`var_smoothing` | 稠密；权重；逐批均值/方差合并；Array API 路径 | `classes_`、`class_count_`、`class_prior_`、`theta_`、`var_`、`epsilon_` |
| `MultinomialNB` | `alpha`、`force_alpha`、`fit_prior`、`class_prior` | 稠密/CSR；权重；增量计数；标量/特征向量平滑 | 类别、计数、对数先验及条件概率 |
| `BernoulliNB` | 上述四项及 `binarize` | 稠密/CSR；阈值或 `None`；权重；增量计数；保留向量平滑的源码广播行为 | 同多项式 |
| `ComplementNB` | 上述四项及 `norm` | 稠密/CSR；权重；增量补集计数；两种归一化；单类别 | 同多项式，另有 `feature_all_` |
| `CategoricalNB` | 上述四项及 `min_categories` | 稠密；标量平滑；每特征类别下限；权重；增量扩展计数表 | `category_count_`、`n_categories_`、按特征组织的 `feature_log_prob_` 等 |

所有模型提供：

- `fit(X, y, sample_weight=None)`，重复拟合重置训练统计量。
- `partial_fit(X, y, classes=None, sample_weight=None)`，首批要求完整类别表，后续检查类别契约。
- `predict`、`predict_joint_log_proba`、`predict_log_proba`、`predict_proba`、加权 `score`。
- `get_params`、`set_params`、`__sklearn_clone__`、文本表示和 `__sklearn_tags__`；可参与常规 `Pipeline`、`GridSearchCV`。
- `get_metadata_routing`、`set_fit_request`、`set_partial_fit_request`、`set_score_request`；支持默认请求、布尔请求、忽略、别名及克隆保留请求。
- 二维、非空、有限实数输入检查；特征数量、字符串特征名和顺序检查；列标签转换警告；构造参数检查；未拟合预测异常。

`n_features_in_` 由训练输入设置；全字符串列名产生 `feature_names_in_`，再次用无列名数据拟合时删除历史列名。CSR 输入保持 CSR；CSC/COO/LIL/DOK/BSR/DIA 等按源码转为 CSR。Gaussian 和 Categorical 拒绝 SciPy 稀疏输入。pandas 全稀疏列走 COO→CSR，而不是先转稠密。

数值计算、标签及属性类型在测试中逐项对照，不仅比较准确率。类别列采用源码的排序顺序；平局按首个最大得分的类别返回。

## 3. 调用示例

```python
from scipy.sparse import csr_array
from NB.sklearn_impl import MultinomialNB

model = MultinomialNB(alpha=0.5)
model.partial_fit(
    csr_array([[3, 1], [0, 4]]), ["垃圾", "正常"],
    classes=["垃圾", "正常"], sample_weight=[2.0, 1.0],
)
model.partial_fit(csr_array([[1, 2]]), ["正常"])
labels = model.predict(csr_array([[2, 0]]))
probability = model.predict_proba(csr_array([[2, 0]]))
```

本地配置与 sklearn 配置相互独立：

```python
from NB.sklearn_impl import GaussianNB, config_context

# xp 是支持 __array_namespace__ 的 Array API namespace。
with config_context(array_api_dispatch=True):
    model = GaussianNB().fit(xp.asarray(X), xp.asarray(y))
```

配置还包括 `assume_finite`、`skip_parameter_validation`、`enable_metadata_routing`、`print_changed_only`。默认值与目标源码一致；配置上下文退出后恢复原状态。元数据请求设置需要启用本地 `enable_metadata_routing`；如果由 sklearn 组合器路由，还需要启用该组合器的 sklearn 配置。普通 `fit(..., sample_weight=...)` 不要求启用元数据路由。

## 4. 保留的边界行为

这些属于源码行为，没有额外加上前两阶段的定义域限制。

- **增量类别：** `classes` 排序后比较。Gaussian 对批次中未知标签报错；离散模型遵循 `label_binarize`，多分类忽略未知标签的指示项，二分类补列后可能归入第一类，单类别指示列被设为 1。没有自行统一成“未知标签一律报错”。
- **高斯平滑：** 每批按当前输入的未加权方差计算 `epsilon_`；第二批开始，源码减去的是本批重新计算的平滑量，再合并统计量并加回该值。保持这一顺序，不能把它改为减去上批平滑量。非零平滑时，批量训练和一次性训练不保证参数完全相同，因此分别与相同批次的参考模型比较。
- **样本权重：** 允许标量以及一维权重；按照 1.9.1 拒绝全零权重，不强制非负。单个类别的权重和接近零时，Gaussian 保留已有均值和方差。权重类型及 Gaussian `float16` 的权重提升规则按源码处理。
- **先验：** Gaussian 检查长度、和约等于 1、非负；离散模型只检查长度，然后取对数，不额外归一化或禁止负值。补集模型在多类别时不加先验，单类别时加先验。
- **平滑：** 允许零平滑；`force_alpha=False` 对小值发出相同警告并使用下限，不改写参数属性。Bernoulli 的特征向量平滑保留源码对 `class_count_` 的广播行为；某些类别数/特征数的组合会报广播错误，没有擅自修正。
- **伯努利输入：** 严格使用大于阈值；`binarize=None` 按源码直接使用输入，不增加只能为 0/1 的检查。重复 CSR 索引逐存储值二值化，不先合并索引；默认复制输入并删除二值化后的零项。
- **类别编码：** 先按源码转换为整数，再检查非负；有限浮点编码会截断，不增加整数性检查。预测超过已分配范围产生 `IndexError`。新批次扩大计数表，但 `n_categories_` 记录本批计算结果，数值可以缩小；历史表不缩小。
- **退化输入：** 零方差、零平滑、补集单特征归一化等按源码保留 `RuntimeWarning`、`inf` 或 `NaN`，不增加固定下限或改为抛出领域异常。Multinomial/Complement 仅在训练计数时检查非负，预测不另加检查。

对数归一化使用该版本 `_logsumexp` 的最大项计数及 `log1p` 计算顺序，不替换成不同版本的 SciPy 归一化规则。补集输出仍是归一化分类得分，不将其描述为普通生成模型的后验概率。

## 5. 本阶段的资源行为及测量

相对于 `optimized/`，新增的是数据表示与训练状态能力：

- 保留非负计数模型和伯努利模型的 CSR 输入，通过二维稀疏乘法计数及评分，不生成高维稠密输入。标签指示矩阵、类别×特征统计量及预测输出仍是稠密。
- 校验有效有限输入先用求和检查，采用源码的常见 $O(n)$ 时间、$O(1)$ 临时空间路径；输入无需转换时复用数组。异常数据可进入更详细的逐元素检查。
- 伯努利按源码复制输入，保留正反条件掩码；CSR 时复制和掩码规模由存储非零项决定。
- `partial_fit` 只保留累计统计量。训练批次由调用者划分，模型没有自行分块、保存历史样本或添加工作内存策略。
- Categorical 仅按出现的编码上限扩充计数表；依然可能为很大的编码分配很大的表，不添加源码未实现的编码压缩。

本目标没有 `n_jobs`，没有新增线程池或并行参数。基准中的线程限制、预热和计时属于验证条件。

[benchmark.py](benchmark.py) 以同一份数据准备稠密/CSR 输入，分别验证本实现与 sklearn 的计数、得分和预测，再测量八批增量训练的累计计数及最终状态大小。默认设置：1200 个样本、8000 个特征、3 类，密度 0.2%，19200 个存储项；随机种子 42，单线程，预热一次、重复三次取计时中位数。

**存储测量：** 稠密输入 73.24 MiB，CSR 输入 0.224 MiB；两种路径的模型统计量大小相同，多项式/伯努利约 0.366 MiB，补集约 0.427 MiB。模型状态不随累计批次数增长。CSR 不能消除类别×特征参数表的开销。

以下是当前代码的新增分配峰值，单位 MiB；不含预先准备好的输入。完整环境、原始数值与计时见 [benchmark_results.json](benchmark_results.json)。

| 模型 | 本实现稠密 fit / predict | 本实现 CSR fit / predict | sklearn CSR fit / predict |
|---|---:|---:|---:|
| MultinomialNB | 0.77 / 0.030 | 0.77 / 0.194 | 0.77 / 0.194 |
| BernoulliNB | 91.56 / 30.52 | 1.00 / 0.64 | 1.00 / 0.64 |
| ComplementNB | 0.83 / 0.017 | 0.83 / 0.194 | 0.83 / 0.194 |

**解释：** 伯努利的稠密复制随样本×特征规模增长，稀疏路径按存储项规模处理，测量验证了这项实际收益。多项式和补集在稠密路径本来就不复制完整输入，CSR 主要节省输入存储，稀疏预测反而可能需要更多临时空间；不能宣称所有路径都降低新增分配峰值或耗时。

内存使用 `tracemalloc`，覆盖可追踪的 Python/NumPy/SciPy 新增分配，不等于进程 RSS 或所有原生库内存。计时与内存采样分开；记录只反映本次环境，不作为普遍加速结论。测试额外禁止对高维 CSR 输入调用 `toarray`；标签编码所需的小指示矩阵仍按源码生成。

## 6. 验证与运行

[test_nb.py](test_nb.py) 共 36 项测试，包含大量模型、类型和参数子测试，覆盖：

- 手算示例、digits；整数/字符串/布尔标签；`int32`、`int64`、`float16`、`float32`、`float64` 输入；标量和数组权重。
- 所有参数选项、统计量/属性类型、四种预测输出、排序与平局、重复拟合、不同增量批次。
- 多种稀疏格式及数组/矩阵、重复索引、显式零、只读输入、复制控制、不允许输入稠密化。
- 类别表增长、浮点编码截断、向量平滑的源码边界、全零或负权重、未知标签、退化数据、警告和异常。
- 克隆、序列化、参数管理、组合器及网格搜索；本地配置和元数据请求/别名路由。
- pandas 列名、nullable 整数和稀疏列；Gaussian Array API 的类型、设备、标签和权重。
- 静态导入审查，以及在新进程阻止所有 sklearn 导入后训练和预测。

浮点通常使用 `rtol=1e-10, atol=1e-12`；单精度使用 `rtol=1e-6`。零平滑等退化边界同时对照 `NaN`/无穷位置及警告类别和正文；正常标签与类别排序精确比较。参考模型独立接收原始输入，不将本实现的预处理结果传给参考模型。

本次扩展依赖临时安装在 `/tmp/nb-test-deps`，没有修改 `DLdemo` 环境。包含 pandas 和 Array API 验证时，本目录 36 项、既有 NB 19 项和 KNN 6 项，**合计 61 项通过**。默认环境缺少这两个可选包时，对应两项测试跳过。

`verify_upstream.py` 读取安装版本的 `sklearn/tests/test_naive_bayes.py`，在临时目录仅替换模型导入，并用配置桥同时设置参考工具和本地模型配置，不改变测试体或断言。采用种子 42，覆盖单/双精度 fixture。本次 **99 项通过、48 项跳过、0 项失败**；跳过的是未安装的 GPU/其他数组后端，不计为已经验证。

公共 estimator 检查分别通过 Gaussian 61 项、Multinomial 61 项、Categorical 60 项、Bernoulli 60 项、Complement 61 项；未通过项及源码自身的相同问题见下一节和 [verification_results.json](verification_results.json)。保存了原始测试文件的 SHA-256，升级版本时需先审核源码差异。

在项目根目录运行默认验证和演示：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.sklearn_impl.main
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m unittest NB.sklearn_impl.test_nb NB.optimized.test_nb NB.basic.test_nb KNN.test_knn
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m NB.sklearn_impl.benchmark --repeats 3 --threads 1
```

扩展验证依赖可安装到临时目录；不需要为运行模型安装这些包：

```bash
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && python -m pip install --target /tmp/nb-test-deps -r NB/sklearn_impl/requirements-test.txt
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && PYTHONPATH=/tmp/nb-test-deps:$PWD SCIPY_ARRAY_API=1 python -m unittest NB.sklearn_impl.test_nb NB.optimized.test_nb NB.basic.test_nb KNN.test_knn
source /home/fire/miniconda3/etc/profile.d/conda.sh && conda activate DLdemo && PYTHONPATH=/tmp/nb-test-deps:$PWD SCIPY_ARRAY_API=1 python -m NB.sklearn_impl.verify_upstream
```

digits 与上层示例采用相同拆分，五种模型的 `fit` / 四批 `partial_fit` 准确率分别为 82.22%、87.59%、90.00%、87.41%、79.44%；各自与同设置、同批次的 sklearn 预测、得分和概率一致。

## 7. 尚未完全复现的行为

以下限制明确保留在验收范围中，不能因为原始模型测试通过就宣称完整兼容：

1. **类对象身份：** `NotFittedError`、`DataConversionWarning`、`InvalidParameterError`、元数据异常和标签 dataclass 是本地实现。名称、常见内容、字段及继承语义对照，但不等于 sklearn 的同名类对象，因此 sklearn 的 `check_valid_tag_types`、`check_estimators_unfitted`、`check_supervised_y_2d` 身份检查不通过。捕获异常应使用本目录导出的类型。普通组合器已验证，要求 sklearn 具体类身份的工具没有完整兼容。
2. **库级框架：** 本地配置不接管 sklearn 的全局配置；HTML estimator 展示、sklearn 回调管理、跨版本 pickle 警告和完整全库配置系统未实现。元数据只实现 NB 简单消费者，不手动重写整个 sklearn 路由器。
3. **输入生态和消息：** pandas 的常用数值/nullable/稀疏路径及列名已验证；Narwhals 支持的所有 dataframe 后端没有全部验证。常见错误类别、参数错误和模型内部错误正文已对照，部分通用校验错误的长提示和格式不同，不保证每个异常逐字符相同。
4. **Array API 后端：** 已验证协议输入的 Gaussian 和 `array-api-strict` 的两个模拟设备；尚未实现/验证所有 PyTorch、CuPy、DPNP namespace 适配及硬件后端。对应跳过项不表示支持已经验收。
5. **源码已有问题：** Gaussian 从 `array-api-strict` 拟合后用 NumPy 预测，会在 `check_array_api_same_namespace` 中因索引 namespace 不匹配而抛出 `IndexError`；参考 sklearn 1.9.1 出现相同错误，JSON 保存了两者消息。没有自行加入该版本 Gaussian 源码未调用的 namespace 检查。

模型计算和主要 NumPy/SciPy 能力已经逐项落地；这些公共框架差异及后端缺口仍需后续处理，第三阶段整体保持“未完全复现”的状态。
