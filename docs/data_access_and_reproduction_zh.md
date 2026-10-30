# 数据访问与复现说明

## 本仓库包含什么

公开内容包括最终主图与PR图、模型结构图、图注、中文方法、汇总性能指标、结局人数/事件比例，以及按特征组汇总的置换重要性与置信区间。表格均为队列层面的汇总值，不含逐参与者观测或预测。

## 为什么不公开完整训练数据

模型训练和个体曲线重绘需要UK Biobank受控访问数据，包括参与者肺功能及临床记录、基因型剂量/PLINK文件、变异schema与效应权重，以及参与者划分和个体预测。逐参与者预测、EID、基因型矩阵、训练折结果、嵌入和模型权重均未纳入本仓库。需要复现的研究者应先取得相应数据授权，并在本地安全环境运行。

## 主要输入文件

实际文件名应依据获批数据导出与原始项目保存的schema核对。代码所需数据至少包括：

| 分支 | 输入内容 | 备注 |
| --- | --- | --- |
| Healthy | `participants_targets_splits.csv.gz` | 参与者内部ID、六项结局、随访间隔和既定train/valid/test划分；仅可在授权环境使用 |
| Healthy | `clinical_features.csv.gz` | 16项临床变量及参与者内部ID |
| Healthy | PLINK `.bed/.bim/.fam`、`genome_variant_schema.csv` | 基因剂量与变异顺序/等位基因对应关系 |
| COPD | `participants_targets_splits.csv.gz`、`fusion_model_features.csv.gz` | COPD参与者结局、随访间隔及既定临床特征 |
| COPD | `genome_variant_schema.csv`、`genome_inputs.npz` | 固定遗传面板与剂量矩阵 |
| COPD | `clinical_models.joblib`、`preprocessing.joblib` | 历史临床基准模板与预处理参数；须来自原始运行产物 |
| 两队列 | 临床字段字典、PRS权重、等位基因方向和基因组版本 | 需与获批输入一致；应补充记录数据版本及QC流程 |

COPD代码当前以原项目保存的临床模型模板及预处理对象为输入，不能仅凭公开汇总表重新构建这些对象。Healthy脚本中的遗传schema也必须与原始面板的变异顺序及等位基因方向一致。

## 本地运行

1. 先取得并整理上表中的授权文件；不得把原始文件复制进本Git仓库。
2. 安装 `requirements.txt` 中的依赖。原始环境的精确Python、PyTorch及scikit-learn版本尚未完整记录，因此需在复现日志中保存实际版本。
3. Healthy脚本使用环境变量 `FEV1_PROJECT_ROOT` 指向授权项目根目录，该目录还需包含代码注释中约定的Healthy、COPD/ECOPD及遗传数据树。示例：

   ```bash
   export FEV1_PROJECT_ROOT=/secure/path/to/authorized/project
   python src/healthy_genome_six_task.py
   ```

4. COPD脚本通过 `--source-root` 接收授权数据和历史模型工件所在目录；例如：

   ```bash
   python src/copd_run_loocv.py --cohort UKB --source-root /secure/path/to/authorized/source
   ```

   `--source-root` 下的目录结构需包含相应队列子目录，以及代码所读取的 `data/` 和 `models/` 文件。请先依据本地项目版本核实字段和文件结构。LOOCV会生成逐参与者折结果、模型检查点和审计文件，应写入受控存储。

上述示例展示必要的配置入口，不代表未经数据准备即可运行。源脚本包含大型队列计算，运行前应由有权限的研究人员核对样本划分、遗传变异对齐、基线COPD定义、伦理/UKB审批、数据QC和所用估计器模板。

## 可由本仓库直接核对的结果

- `data/aggregate/performance_bootstrap_95CI.csv`：六任务性能及参与者bootstrap区间。
- `data/aggregate/outcome_distributions.csv`：任务样本量、事件数及比例。
- `data/aggregate/group_permutation_importance_95CI.csv`：基线肺功能、随访间隔、年龄、吸烟和遗传组的置换重要性。
- `data/aggregate/exploratory_group_differences.csv`：探索性组间差异；不能单独证明疾病特异性机制。
- `results/continuous_metrics_Pearson_Spearman_R2.csv`：三个连续任务的Pearson、Spearman和R²。

当前最终散点、ROC与PR图的重绘需要受控个体预测；仓库提供的是最终图件和对应汇总表，不含绘制这些曲线所需的参与者级数据。

## Reproduction acceptance status

The standalone PR figure and AP/AUPRC aggregate values are excluded from this public release. The synthetic demo contains invented values for input-schema and plotting checks only, not real test-set observations or model predictions.

Paper-performance reproduction remains unverified. It requires authorized original splits, real labels and predictions, or training data plus the original preprocessing/model artifacts and environment versions. Evaluate with identical task definitions, sample counts and cohort evaluation designs; compare the resulting metrics with the paper and record discrepancies. Retraining and metric recomputation from fixed predictions must be assessed separately. Public aggregate tables and synthetic examples cannot establish reproduced performance. Metric definitions and evaluation code are retained for authorized evaluation.
