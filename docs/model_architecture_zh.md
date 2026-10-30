# 基于原训练代码的模型建立与绘图说明

这是一套临床基准预测加基因组Transformer残差修正框架；Healthy和COPD是分别建模的人群，不是二分类标签。当前图仅解释两套UKB主模型，不把广东token适配分支混入源模型结构。

## 数据与六个目标

输入为基线临床信息、计划/实际预测间隔、4,064个固定遗传位点的等位基因剂量，以及由这些剂量和GWAS权重构建的四个PRS。PRS采用P≤1e-5、1e-4、1e-3、0.01四个阈值，相互重叠，不是独立遗传信息。缺失填补和标准化参数只由相应训练数据估计。遗传面板是历史COPD发现的固定面板；不能把这一流程画成每一外层折都重新独立GWAS发现。

六项监督目标为：下一次实际FEV₁、FVC、FEV₁/FVC；以及三项快速下降事件。容量快速下降为(next-baseline)/interval<-0.060 L/年，比值快速下降为年变化<-0.010/年。风险头直接受事件标签监督，不是把连续预测值用阈值转换而来；更不是预测Healthy/COPD疾病状态。

## 公共基因组编码器

按染色体分组，按schema既定顺序把每64个SNP剂量分为一个patch，尾部补零。线性投影64→16，与染色体嵌入、物理位置编码相加。位置特征为均值位置/1e8及其sin、cos、log1p，经4→16线性投影。

Transformer仅1层，2个注意力头，嵌入16维，前馈32维，dropout0.35，norm_first=True。不能照参考图写8层或8头。

四个PRS各自做1→16线性投影，再加可学习的PRS类型嵌入；它们在SNP self-attention之后才与SNP token拼接，经LayerNorm进入任务query注意力汇聚。query对token作softmax权重并加权求和，为每项任务生成16维h_j。这里是任务注意力池化，不是额外的完整跨注意力Transformer层。注意力权重不是已经验证的特征重要性。

## Healthy模型

16项临床字段为Age_BL、BMI_BL、Height_BL、Pack_years_BL、FEV1_BL、FVC_BL、FEV1pred%_BL、FEV1_FVC_RATIO_BL、Sex、SMK_status_BL、p30000_i0、p30120_i0、p30130_i0、p30140_i0、p30150_i0和interval_years。

先分别拟合三项Ridge连续基准和三项logistic分类基准；中位数插补、缺失指示、标准化纳入管线，超参数在固定验证集选择。训练者的三折临床折外预测用于构建残差目标及分类logit offset，减少直接用同一样本拟合残差的乐观性。

神经分支包含上述基因组编码器和额外临床MLP（变换后临床维度→32→16）。每项任务把16维h_j与16维临床向量拼为32维，送入32→16→1任务头；六个末层均从零初始化，使起点对应临床基准。

三个回归头训练标准化临床残差，三个分类头训练clinical logit offset加神经修正后的BCE，按有效标签mask汇总为六任务均值。训练采用AdamW、梯度裁剪，最多60轮，验证损失patience10早停；验证集选择checkpoint及各任务w∈{0,0.1,0.25,0.5,1}。连续输出为b_j+w_j s_j δ_j；分类输出为sigmoid(logit(b_j)+w_j δ_j)。固定测试集1,899人。当前Healthy为单种子模型，不应标成两种子集成。

## COPD模型

临床基准为各任务既定临床输入加31个lead SNP；回归使用仿射Ridge管线，分类使用原logistic基准管线。年龄平方、身高平方等沿用原任务特征，不能直接把Healthy的16字段数量写到COPD临床分支。

外层逐人留一；临床模型、折内基因组/PRS归一化及梯度训练均排除该参与者。外层训练部分的三折临床折外预测用于构造神经训练目标。

与Healthy不同，当前UKB COPD的AlignedGenome神经修正分支不接临床MLP；临床信息通过基准旁路进入最终融合。基因组网络的回归与分类分开训练，各有三个query及三个任务头。两种子分别为20261026、20261027；每个外层模型训练固定60轮，回归和分类分别做两种子均值。连续残差按折内均值μ和标准差s标准化，最终输出b_j+μ_j+s_j δbar_j；分类为sigmoid(logit(b_j)+δbar_j)，神经残差权重固定为1。

回归网络为三任务masked MSE。分类网络为三任务clinical-offset BCE，按训练事件率计算的熵作任务归一化。不能把COPD画成与Healthy完全相同的六头单网络联合训练流程。最终留一折外预测2,067人。固定历史面板、架构和超参数的LOOCV，并不等同于每折重新开展GWAS/特征发现的完全嵌套验证。

## 参考图的视觉元素如何替换

scRNA-seq矩阵→基因型剂量矩阵；gene embeddings→SNP patch embeddings；8×encoder blocks→1×encoder block、2 attention heads；cell embeddings→task-specific genomic embeddings；clinical feature→基线临床向量与单独的临床基准旁路；individual embeddings→Healthy的临床-基因组拼接或COPD的基因组任务表示；FEV1%pred回归标签→三项下一次实际肺功能；disease status→三项快速下降标签；importance matrices→独立的置换重要性/积分梯度解释分析。

验证集或留一外层患者应画在训练框外，用评估箭头表示。反向传播箭头只指向当前训练网络，不能画成从测试结局返回训练。临床基准与神经残差分阶段拟合，不是整个临床Ridge/logistic和Transformer全部端到端共同更新。

图中A为公共编码模块定义，B和C为各自实例化的完整模型，不表示Healthy和COPD共享已训练权重。图中的解释性分析是建模完成后的附加步骤，不应把attention heatmap直接标为临床或遗传因果机制。

代码依据：source/healthy_genome_six_task.py（Fusion、clinical_fit、train_net、main）；source/copd_run_loocv.py（oofclinical、trainnet、外层循环）；source/aligned.py（AlignedGenome）。附带的源代码副本仅用于架构核对，原模型及权重未修改。
