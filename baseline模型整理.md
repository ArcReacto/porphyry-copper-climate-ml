下面是我建议你第一版 baseline 使用的模型清单。整体原则是：**先用成熟的表格数据模型，不上深度学习，不用预训练模型。**

| 优先级 | 模型 | 开源实现/链接 | 使用目的 | 是否第一版使用 |
|---|---|---|---|---|
| 1 | DummyClassifier | [scikit-learn DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) | 最低基线。它不看任何特征，只按简单规则预测，用来回答“模型至少要比瞎猜好多少”。官方也说明它用于和复杂分类器比较。 | 必用 |
| 2 | Logistic Regression | [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | 线性可解释 baseline。看地球化学/重力特征是否能通过简单线性关系区分正负样本。 | 必用 |
| 3 | Random Forest | [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | 第一版主力模型。适合表格数据、非线性关系、特征较多的情况，还能输出特征重要性。 | 必用 |
| 4 | HistGradientBoosting | [scikit-learn HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) | scikit-learn 自带 boosting 模型，不用额外装 XGBoost/LightGBM。可作为 Random Forest 的增强对照。 | 建议用 |
| 5 | XGBoost | [XGBoost Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html) / [XGBoost docs](https://xgboost.readthedocs.io/) | 强表格数据 boosting baseline。适合后续和 Random Forest 比较，但样本量不大时要防过拟合。 | 第二版可加 |
| 6 | LightGBM | [LightGBM LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) | 快速梯度提升模型。适合高维表格特征，后续特征更多时很有用。 | 第二版可加 |
| 7 | Permutation Importance | [scikit-learn permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) | 不是分类模型，而是解释工具。用来检查模型到底依赖 Cu、Mo、As、重力，还是依赖数据覆盖类字段。 | 必用解释工具 |

**第一版我建议实际跑这 4 个就够：**

```text
DummyClassifier
LogisticRegression
RandomForestClassifier
HistGradientBoostingClassifier
```

原因是它们都在 `scikit-learn` 里，你当前环境基本已经有，不需要额外装复杂依赖。

**第二版再加：**

```text
XGBoost
LightGBM
```

这两个要额外安装：

```powershell
.\.venv\Scripts\python.exe -m pip install xgboost lightgbm
```

**每个模型在你项目里的具体作用**

```text
DummyClassifier
判断模型有没有超过“瞎猜/多数类”的最低水平。

Logistic Regression
判断是否存在简单、稳定、可解释的线性找矿信号。

Random Forest
作为第一版主模型，捕捉非线性地化/重力组合特征。

HistGradientBoosting
作为 scikit-learn 内置 boosting 对照，检查 boosting 是否明显优于随机森林。

XGBoost / LightGBM
作为后续增强模型，用于更强表格预测能力对比。

Permutation Importance
解释哪些元素和重力特征真正影响预测，避免模型学到采样覆盖偏差。
```

**对应你的数据集，我建议第一版这样跑**

```text
数据集：
outputs/model_features_southwest_core.parquet
outputs/model_features_western_core.parquet

第一版特征：
geochem1_usgs_*
gravity_na_*

第二版敏感性分析：
geochem1_usgs_*
geochem2_nure_*
gravity_na_*
```

**评估方式要配套**

模型列表之外，还要同时做：

```text
StratifiedKFold
GroupKFold by state
按 negative_type 分组评估
```

这样能看出模型是真有找矿信号，还是只是在学州差异、负样本类型差异、数据覆盖差异。

如果你确认这个列表没问题，下一步我就可以开始写：

```text
scripts/09_make_model_dataset.py
scripts/10_train_baseline.py
```