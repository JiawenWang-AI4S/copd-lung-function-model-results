from pathlib import Path
import os, tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'lung_model_mpl'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'figures';T=['FEV1_next','FVC_next','ratio_next','interval_rapid_decline','interval_FVC_rapid_decline','interval_ratio_rapid_decline'];L=['FEV₁','FVC','FEV₁/FVC'];C=['#4C91B2','#DF8B72','#4BA99D'];PC={'Healthy':'#4C91B2','COPD':'#DF8B72'};D={p:pd.read_csv(R/f'data/local_inputs/{p}_selected_predictions.csv.gz') for p in PC};M=pd.read_csv(R/'data/aggregate/performance_bootstrap_95CI.csv');PI=pd.read_csv(R/'data/aggregate/group_permutation_importance_95CI.csv');rng=np.random.default_rng(20261028);grid=np.linspace(0,1,201);rows=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
def curves(y,p):
 ix=np.argsort(-p,kind='stable');a=y[ix];v=p[ix];ends=np.r_[np.flatnonzero(np.diff(v)),len(a)-1];tp=np.cumsum(a)[ends];fp=ends+1-tp;return np.r_[0,fp/(len(a)-sum(a))],np.r_[0,tp/sum(a)],np.r_[0,tp/sum(a)],np.r_[1,tp/(tp+fp)]
def score(pop,t,metric):return M[(M.population==pop)&(M.task==t)&(M.metric==metric)].iloc[0]
f=plt.figure(figsize=(10.8,20.2));gs=f.add_gridspec(6,2,left=.17,right=.96,bottom=.095,top=.935,hspace=.43,wspace=.65,height_ratios=[1,1,1,1,1,1]);f.text(.055,.975,'Healthy and COPD | performance and model interpretation',weight='bold',size=16);f.text(.29,.952,'HEALTHY  ·  fixed test, n = 1,899',ha='center',size=12,color=PC['Healthy']);f.text(.78,.952,'COPD  ·  LOOCV, n = 2,067',ha='center',size=12,color=PC['COPD'])
prfig,prax=plt.subplots(1,2,figsize=(12,5.9));prfig.subplots_adjust(left=.085,right=.97,top=.79,bottom=.21,wspace=.27);prfig.text(.055,.95,'Three rapid-decline tasks | precision–recall comparison',weight='bold',size=16)
for col,(pop,d) in enumerate(D.items()):
 for k,t in enumerate(T[:3]):
  ax=f.add_subplot(gs[k,col]);q=d[d.task==t];y=q.observed.to_numpy();p=q.predicted.to_numpy();mu=y.mean();r=np.corrcoef(y,p)[0,1];rho=np.corrcoef(pd.Series(y).rank(),pd.Series(p).rank())[0,1];r2=1-((y-p)**2).sum()/((y-mu)**2).sum();rows.append(dict(population=pop,task=t,n=len(y),Pearson=r,Spearman=rho,R2=r2))
  allvals=np.concatenate([D[g].loc[D[g].task==t,['observed','predicted']].to_numpy().ravel() for g in D]);lo=allvals.min();hi=allvals.max();pad=(hi-lo)*.06;lo-=pad;hi+=pad
  ax.scatter(y,p,s=7,color=C[k],alpha=.55,edgecolors='none',rasterized=True);ax.plot([lo,hi],[lo,hi],'--',color='#94A1AE',lw=.8)
  ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed'+(' (L)' if k<2 else ' ratio'),ylabel='Predicted'+(' (L)' if k<2 else ' ratio'));ax.set_aspect('equal',adjustable='box');ax.set_title(f'{chr(65+col)}{k+1}  {L[k]}',loc='left',fontsize=10,pad=10);ax.tick_params(labelsize=7)
  ax.text(.035,.965,f'Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nR² = {r2:.3f}',transform=ax.transAxes,size=7,va='top',linespacing=1.4,bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1))
 roc=f.add_subplot(gs[3,col]);roc.set_title(f'{chr(67+col)}  Three rapid-decline ROC curves',loc='left',pad=12);roc.plot([0,1],[0,1],'--',color='#ADB6BD',lw=.8)
 pr=prax[col];pr.set_title(f'{chr(65+col)}  {pop}',loc='left',pad=12)
 for k,t in enumerate(T[3:]):
  q=d[d.task==t];y=q.observed.to_numpy();p=q.predicted.to_numpy();fp,tp,rec,prec=curves(y,p);roc_bs=[];pr_bs=[]
  for b in range(500):
   ids=rng.integers(0,len(y),len(y));yy=y[ids]
   if yy.sum()==0 or yy.sum()==len(yy):continue
   bx,by,br,bp=curves(yy,p[ids]);roc_bs.append(np.interp(grid,bx,by));pr_bs.append(np.interp(grid,br,bp))
  lo,hi=np.quantile(roc_bs,[.025,.975],axis=0);roc.fill_between(grid,lo,hi,color=C[k],alpha=.14,lw=0);a=score(pop,t,'AUROC');roc.plot(fp,tp,color=C[k],lw=1.6,label=f'{L[k]}  AUROC {a.estimate:.3f}')
  lo,hi=np.quantile(pr_bs,[.025,.975],axis=0);pr.fill_between(grid,lo,hi,color=C[k],alpha=.17,lw=0);pr.plot(rec,prec,color=C[k],lw=1.8,label=f'{L[k]}  events {int(y.sum())}/{len(y)}')
 roc.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive rate',ylabel='True-positive rate');roc.set_box_aspect(1);roc.legend(loc='lower right',fontsize=8,frameon=False)
 pr.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision');pr.legend(loc='upper right',fontsize=8,frameon=True,facecolor='white',edgecolor='none',framealpha=.94)
 ax=f.add_subplot(gs[4,col]);ax.set_title(f'{chr(69+col)}  Clinical Top 10 | next FEV₁',loc='left',pad=12)
 a=pd.read_csv(R/f'results/{pop}_clinical_top10.csv');yy=np.arange(len(a));labels={'FEV1_BL':'Baseline FEV₁','FVC_BL':'Baseline FVC','FEV1pred%_BL':'Baseline FEV₁ (% predicted)','FEV1_FVC_RATIO_BL':'Baseline FEV₁/FVC','interval_years':'Follow-up interval','Age_BL':'Age','Age_square_BL':'Age squared','Sex':'Sex','Height_BL':'Height','Height_square_BL':'Height squared','Pack_years_BL':'Smoking pack-years','SMK_status_BL':'Smoking status','BMI_BL':'BMI'}
 ax.barh(yy,a.mean_abs_contribution_L,color=PC[pop],height=.66);ax.set(yticks=yy,yticklabels=[labels.get(v,'Blood count ('+v+')' if v.startswith('p30') else v) for v in a.feature],xlim=(0,.55),xlabel='Mean absolute clinical contribution (L)');ax.tick_params(axis='y',labelsize=7)
 for y0,v in zip(yy,a.mean_abs_contribution_L):ax.text(v+.008,y0,f'{v:.3f}',va='center',size=7)
 a.assign(population=pop).to_csv(O/f'{pop}_clinical_top10.csv',index=False)
 ax=f.add_subplot(gs[5,col]);ax.set_title(f'{chr(71+col)}  Direct SNP-channel IG | next FEV₁',loc='left',pad=12)
 a=pd.read_csv(R/f'results/{pop}_direct_SNP_top10.csv');v=a.value.to_numpy()*1000;ax.barh(range(10),v,color=PC[pop],height=.66);ax.set_yticks(range(10),[v if len(v)<=23 else v[:15]+'…'+v[-5:] for v in a.variant]);ax.tick_params(axis='y',labelsize=6.8);ax.set(xlabel='Mean absolute direct SNP-channel IG (mL)',xlim=(0,2.85))
 r=PI[(PI.population==pop)&(PI.group=='All_genetics_joint')].iloc[0];ax.text(.02,-.28,f'Joint genetics + PRS ΔRMSE: {r.delta_RMSE:.5f} L\n95% conditional CI [{r.ci_low:.5f}, {r.ci_high:.5f}]',transform=ax.transAxes,fontsize=8)
f.text(.055,.047,'Task colors: FEV₁ (blue), FVC (coral), FEV₁/FVC (teal). A1–A3 and B1–B3: separate tasks in original units; matched axes by task.',size=8)
f.text(.055,.034,'C–D bands: 500 patient bootstraps. E–F: native clinical attribution Top 10; Healthy linear + neural IG, COPD affine branch.',size=8)
f.text(.055,.021,'G–H show direct SNP input-channel IG only; PRS and clinical-SNP paths excluded. Different designs / inputs limit specificity claims.',size=8)
prfig.text(.085,.125,'Bands: pointwise 95% intervals from 500 participant bootstraps of saved predictions; no model refitting.',size=9);prfig.text(.085,.09,'Healthy fixed test; COPD LOOCV. These curves predict decline within each cohort, not COPD diagnosis.',size=9);prfig.text(.085,.055,'Rapid decline: annual change < -0.060 L/year (FEV₁, FVC), or < -0.010/year (FEV₁/FVC).',size=9)
for fig,name in [(f,'Figure1_integrated_two_columns'),(prfig,'Figure2_three_task_PR_curves')]:
 for ext in ['pdf','png','svg']:fig.savefig(O/f'{name}.{ext}',dpi=300)
 plt.close(fig)
pd.DataFrame(rows).to_csv(R/'results/continuous_metrics_Pearson_Spearman_R2_recomputed.csv',index=False);print(pd.DataFrame(rows).to_string(index=False));print('COMPLETE')
