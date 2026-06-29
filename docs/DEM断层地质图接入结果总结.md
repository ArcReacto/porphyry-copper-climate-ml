# DEM、断层、地质图接入结果总结

本文档总结当前已经接入的三类新增空间特征：DEM 地形、断层和 CMMI 地质图。

## 1. 接入数据源

### 1.1 DEM

原始数据位置：

```text
C:\Users\PC\Desktop\探矿气象数据集\DEM
```

当前使用数据：

```text
SRTMGL1 30m HGT tiles
```

检查结果：

```text
HGT 文件数: 265
western_core 所需瓦片缺失: 0
```

### 1.2 断层数据

原始数据：

```text
C:\Users\PC\Desktop\探矿气象数据集\地质\CMMI_Faults\GeologyFaults_USCanada\GeologyFaults_USCanada.shp
```

数据结构：

```text
类型: POLYLINE
记录数: 7602
字段: Fault_ID
```

### 1.3 CMMI 地质图

原始数据：

```text
C:\Users\PC\Desktop\探矿气象数据集\地质\CMMI_Geology\Geology_CONUS\Geology_CONUS.shp
C:\Users\PC\Desktop\探矿气象数据集\地质\CMMI_Geology\CMMI_Classification.csv
```

数据结构：

```text
类型: POLYGON
记录数: 313732
主要字段: UNIT_LINK, CMMI_Class, UNIT_NAME, Shape_Leng, Shape_Area
```

## 2. 新增脚本

新增脚本：

```text
C:\Users\PC\Desktop\探矿气象项目代码\scripts\11_align_terrain_geology.py
```

该脚本在 `05_spatial_align_features.py` 之后运行，读取已有：

```text
outputs\model_features_samples.parquet
```

然后追加：

```text
terrain_*
fault_*
geology_*
```

追加完成后覆盖更新：

```text
outputs\model_features_samples.parquet
outputs\model_features_samples.csv
```

完整管线 `run_pipeline.ps1` 已经加入该步骤。

## 3. 新增字段

### 3.1 DEM 地形字段

| 字段 | 含义 |
|---|---|
| `terrain_tile` | 样本点所在 HGT 瓦片 |
| `terrain_has_dem` | 是否找到 DEM 瓦片，1 表示有 |
| `terrain_elevation_m` | 样本点最近像元高程 |
| `terrain_slope_deg` | 基于邻近像元估算的坡度 |
| `terrain_roughness_3x3_m` | 3x3 邻域内高程极差 |
| `terrain_relief_250m_m` | 250 m 邻域局部起伏 |
| `terrain_relief_1000m_m` | 1000 m 邻域局部起伏 |
| `terrain_relief_5000m_m` | 5000 m 邻域局部起伏 |

### 3.2 断层字段

| 字段 | 含义 |
|---|---|
| `fault_nearest_distance_km` | 样本点到最近断层的距离 |
| `fault_lines_5km` | 5 km 内相交断层条数 |
| `fault_length_km_5km` | 5 km 内断层总长度 |
| `fault_lines_10km` | 10 km 内相交断层条数 |
| `fault_length_km_10km` | 10 km 内断层总长度 |
| `fault_lines_25km` | 25 km 内相交断层条数 |
| `fault_length_km_25km` | 25 km 内断层总长度 |
| `fault_lines_50km` | 50 km 内相交断层条数 |
| `fault_length_km_50km` | 50 km 内断层总长度 |

### 3.3 CMMI 地质图字段

| 字段 | 含义 |
|---|---|
| `geology_class_found` | 是否匹配到 CMMI 地质面 |
| `geology_CMMI_Class` | 样本点所在地质单元的 CMMI 分类 |
| `geology_UNIT_NAME` | 原始地质单元名称 |
| `geology_UNIT_LINK` | 原始地质单元链接/编号 |
| `geology_is_igneous` | 是否为 igneous 类 |
| `geology_is_intrusive` | 是否为 intrusive 类 |
| `geology_is_volcanic` | 是否为 volcanic 类 |
| `geology_is_felsic` | 是否为 felsic 类 |
| `geology_is_intermediate` | 是否为 intermediate 类 |
| `geology_is_mafic` | 是否为 mafic 类 |
| `geology_is_sedimentary` | 是否为 sedimentary 类 |
| `geology_is_carbonate` | 是否为 carbonate 类 |
| `geology_is_metamorphic` | 是否为 metamorphic 类 |
| `geology_is_unconsolidated` | 是否为 unconsolidated 类 |

## 4. 接入后特征表规模

接入前：

```text
model_features_samples: 1089 行, 555 列
```

接入后：

```text
model_features_samples: 1089 行, 586 列
western_core: 474 行, 586 列
southwest_core: 306 行, 586 列
```

新增字段数：

```text
terrain/fault/geology 新增字段合计: 31
```

其中进入建模数据集的数值字段约为：

```text
southwest_core: 22 个新增数值特征
western_core: 23 个新增数值特征
```

## 5. 覆盖情况

### 5.1 全域样本表

```text
model_features_samples:
DEM 覆盖: 474 / 1089
CMMI 地质图覆盖: 507 / 1089
```

全域表包含 Alaska、Puerto Rico 和其他非 western_core 样本，因此 DEM 和 CONUS 地质图不全覆盖是预期结果。

### 5.2 Western Core

```text
western_core:
DEM 覆盖: 474 / 474
CMMI 地质图覆盖: 474 / 474
```

### 5.3 Southwest Core

```text
southwest_core:
DEM 覆盖: 306 / 306
CMMI 地质图覆盖: 306 / 306
```

结论：

```text
western_core 和 southwest_core 的 DEM、断层、地质图覆盖完整，可以作为后续主分析区特征。
```

## 6. 接入后的模型影响

新增三类数据后，额外生成了两类建模数据集：

```text
terrain_geo = geochem1 + gravity + DEM + faults + geology
full        = geochem1 + geochem2 + gravity + DEM + faults + geology
```

关键结果：

| 对比 | 结果 |
|---|---|
| `southwest_core_primary_v1` -> `southwest_core_terrain_geo_v1` | Random Forest ROC-AUC 0.971 -> 0.972，F1 0.821 -> 0.850 |
| `western_core_primary_v1` -> `western_core_terrain_geo_v1` | Random Forest ROC-AUC 0.945 -> 0.947，HGB F1 0.774 -> 0.790 |
| `western_core_primary_v1` -> `western_core_full_v1` | Logistic ROC-AUC 0.701 -> 0.859 |

解释：

- 新增三类数据没有造成模型异常。
- 对树模型，新增特征带来的是小幅补充，不是主导提升。
- 对 Logistic Regression，新增特征改善更明显，说明 DEM/断层/地质图提供了更线性的区域差异信息。

## 7. 新增特征的重要性

当前最有贡献的新增特征主要来自 DEM 地形：

```text
terrain_relief_250m_m
terrain_relief_1000m_m
terrain_relief_5000m_m
terrain_roughness_3x3_m
terrain_slope_deg
```

断层和地质图有辅助贡献：

```text
fault_length_km_50km
fault_length_km_25km
fault_length_km_10km
geology_is_intrusive
geology_is_intermediate
geology_is_felsic
```

整体判断：

```text
地球化学仍是主信号；
DEM 地形是稳定补充信号；
断层和地质图目前是辅助结构信息；
气象数据接入后，需要重新比较完整模型。
```

## 8. 当前解释边界

1. DEM 特征来自 30m HGT 瓦片，适合表示局部高程、坡度和地形起伏。
2. 断层特征来自 CMMI 断层线数据，当前只使用距离、条数和长度，没有区分断层类型或活动性。
3. 地质图当前使用点位所在的 CMMI 面单元，并将 `CMMI_Class` 转成关键词 0/1 标记。
4. 地质图面数据很大，当前没有计算邻域内地质类型面积占比；后续如需要可继续扩展。
5. 新增特征是表格空间特征，不属于图像或多模态数据，符合当前“不要接入图片等多模态数据”的要求。

## 9. 输出位置

增强后的特征表：

```text
C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_features_samples.parquet
C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_features_western_core.parquet
C:\Users\PC\Desktop\探矿气象项目代码\outputs\model_features_southwest_core.parquet
```

接入日志：

```text
C:\Users\PC\Desktop\探矿气象项目代码\logs\11_align_terrain_geology_summary.json
```

