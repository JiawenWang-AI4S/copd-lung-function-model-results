# 两列合并版图注

主图左列Healthy，右列COPD。每列纵向依次排列三个独立连续任务（FEV₁、FVC、FEV₁/FVC）、ROC、临床Top 10及遗传Top 10，共六行两列。

A1–A3、B1–B3：顶部每组拆为三个独立的观察值-预测值散点图，依次为FEV₁、FVC、FEV₁/FVC，各自标注Pearson r、Spearman ρ及R²（1-SSE/SST），不再叠加三个任务。FEV₁与FVC使用原始L，比值无量纲；同一任务的Healthy与COPD采用相同的横、纵坐标范围并保留全部个体。配色为Nature风格中等明度蓝（#4C91B2）、珊瑚橙（#DF8B72）和青绿（#4BA99D）；散点透明度0.55。临床及遗传条形图沿用队列蓝／珊瑚橙。

C、D：每组同一面板展示三项快速下降ROC曲线，图例标注AUROC。阴影为500次参与者bootstrap、固定模型预测条件下的逐点95%区间，非整条曲线同时置信带，也不包含训练不确定性。

E、F：下一次FEV₁任务各自的临床特征Top 10，直接使用两套原主图的临床贡献数据，按平均绝对贡献（L）降序选取前10位；两列横轴统一为0–0.55 L。Healthy为完整模型临床贡献（精确线性分解加神经网络IG，缺失指示与原字段先有符号合并再求平均绝对值）；COPD为仿射临床分支贡献（按各折训练背景计算、展开项合并回原输入）。年龄平方、身高平方等沿用原图原生特征，不改成分组置换重要性。这两种原生归因口径不同，仅用于各自模型内部解释，不以绝对值差异证明疾病特异性。不为这批原图汇总值补造置信区间。对应CSV保存实际排序与数值。

G、H：下一次FEV₁任务的直接SNP通道Integrated Gradients前10位，均为mL，仅使用直接剂量通道，排除PRS路径和COPD临床分支SNP贡献，避免Healthy直接IG与COPD包含PRS及临床SNP的总归因混用。两列采用相同横轴范围；IG仍受各自模型、参考背景及输入分布影响，不能直接证明疾病特异性。过长变异标签以省略号显示，完整chr:pos:ref:alt保留在各组CSV中。面板下方另列所有遗传路径及PRS联合置换的ΔRMSE及条件95%区间。


Healthy固定测试集1,899人，COPD留一折外预测2,067人。快速下降事件数：FEV₁ 402与443，FVC 560与890，比值77与63。两组评估设计、输入及历史遗传特征发现不一致。所有区间均未覆盖模型训练和上游筛选不确定性；模型依赖差异不等于疾病状态导致的生物学差异。这里的Healthy沿用原研究基线非COPD定义。

数据来源：此前指定的Healthy_Transformer_main_figure与COPD_Transformer_LOOCV_main_figure配套个体预测和归因数据；临床联合置换及遗传／PRS整体置换来自上一轮真实数据补算。没有新训练模型，也没有将参考图的示意曲线或条长充作结果。


排版修订：维持六行两列，ROC使用1:1正方形面板，与连续结局散点图对齐；缩短画布高度并压缩行距。推荐排序为连续结局（FEV₁、FVC、比值）→快速下降ROC→临床Top10→遗传Top10，便于由预测表现进入模型解释，并逐行比较两个人群。

The standalone PR figure is excluded from this release. Authorized PR regeneration labels event counts only; AP/AUPRC summary labels have been removed.
