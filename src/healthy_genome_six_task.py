"""Healthy UKB to baseline non-COPD ECOPD, six supervised endpoints.

Frozen COPD-discovered SNP panel is used as an independent discovery panel for
Healthy UKB. All clinical fits, normalization and checkpoints are train-local.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,copy,sys,random,time
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error,log_loss,balanced_accuracy_score,roc_auc_score,average_precision_score,accuracy_score,r2_score,confusion_matrix
from scipy.special import logit,expit
from scipy.stats import pearsonr,spearmanr

_project_root=os.environ.get('FEV1_PROJECT_ROOT')
if not _project_root:
    raise RuntimeError('Set FEV1_PROJECT_ROOT to the authorized local project data root; participant data are not distributed here.')
ROOT=Path(_project_root).expanduser()
O=ROOT/'model1_healthy_fast_decline/Healthy_six_task_genome_transfer_20261026'
REF=ROOT/'shared_state_prediction_20261018/UKB_GD_reproduction_inference_20261026'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from aligned import AlignedGenome
TASKS=['FEV1_next','FVC_next','ratio_next','interval_rapid_decline','interval_FVC_rapid_decline','interval_ratio_rapid_decline']
FEATURES=['Age_BL','BMI_BL','Height_BL','Pack_years_BL','FEV1_BL','FVC_BL','FEV1pred%_BL','FEV1_FVC_RATIO_BL','Sex','SMK_status_BL','p30000_i0','p30120_i0','p30130_i0','p30140_i0','p30150_i0','interval_years']
CFG=dict(name='small',dim=16,heads=2,layers=1,patch=64,dropout=.35,lr=3e-4,weight_decay=.03)
SEED=20261026
torch.set_num_threads(2)
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'

class Fusion(nn.Module):
    def __init__(self,schema,kind,nclinical):
        super().__init__();self.genome=AlignedGenome(schema,kind,CFG,6)
        self.clinical=nn.Sequential(nn.Linear(nclinical,32),nn.GELU(),nn.Dropout(.35),nn.Linear(32,16))
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Dropout(.35),nn.Linear(16,1)) for _ in TASKS])
        for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def forward(self,g,p,c):
        _,h,_=self.genome(g,p);clinical=self.clinical(c)
        return torch.cat([head(torch.cat([h[:,j],clinical],1)) for j,head in enumerate(self.heads)],1),h

def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

def numeric(x):return x[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(float)

def prepare_ukb():
    t=pd.read_csv(O/'data/participants_targets_splits.csv.gz',dtype={'eid':str})
    x=pd.read_csv(O/'data/clinical_features.csv.gz',dtype={'eid':str}).set_index('eid').loc[t.eid].reset_index()
    s=pd.read_csv(REF/'UKB/data/genome_variant_schema.csv')
    prefix=ROOT/'wangjiawen_FEV1_decline/genotype/allchr.nec3.QC.plink1'
    fam=pd.read_csv(str(prefix)+'.fam',sep=r'\s+',header=None,dtype=str);lut={v:i for i,v in enumerate(fam[1])}
    pos=np.array([lut.get(v,-1) for v in t.eid]);known=pos>=0;width=(len(fam)+3)//4
    bed=np.memmap(str(prefix)+'.bed',dtype='uint8',mode='r',offset=3)
    g=np.full((len(t),len(s)),np.nan,dtype='float32');dec=np.array([2,np.nan,1,0],dtype='float32')
    # Audited schema contains exact BIM row offsets and literal allele pairs.
    for j,r in s.iterrows():
        ii=pos[known];v=dec[(bed[int(r.bed_index)*width+ii//4]>>(2*(ii%4)))&3]
        if str(r.A1)==str(r.Allele2) and str(r.A2)==str(r.Allele1):pass
        elif str(r.A2)==str(r.Allele2) and str(r.A1)==str(r.Allele1):v=2-v
        else:raise ValueError('Allele mismatch '+r.MarkerID)
        g[known,j]=v
    np.savez_compressed(O/'data/healthy_raw_genome.npz',eid=t.eid.to_numpy(str),dosage=g,matched=known)
    s.to_csv(O/'data/genome_variant_schema.csv',index=False)
    t['genotype_matched']=known
    t.to_csv(O/'data/participants_targets_splits.csv.gz',index=False)
    return t,x,g,s

def clinical_fit(t,x,folder):
    folder.mkdir(parents=True,exist_ok=True);Y=t[TASKS].to_numpy(float)
    Y[~t.interval_years.between(.5,13).to_numpy()]=np.nan
    tr=t.split.eq('train').to_numpy();va=t.split.eq('valid').to_numpy()
    B=np.zeros_like(Y);OOF=np.full_like(Y,np.nan);models={};rows=[]
    for j,task in enumerate(TASKS):
        fit=np.flatnonzero(tr&np.isfinite(Y[:,j]));valid=np.flatnonzero(va&np.isfinite(Y[:,j]));best=None
        for param in ([.1,1,10,100] if j<3 else [.001,.01,.1,1]):
            estimator=Ridge(alpha=param) if j<3 else LogisticRegression(C=param,max_iter=2000,random_state=SEED)
            m=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('model',estimator)])
            m.fit(x.iloc[fit],Y[fit,j]);p=m.predict(x.iloc[valid]) if j<3 else m.predict_proba(x.iloc[valid])[:,1]
            loss=mean_squared_error(Y[valid,j],p) if j<3 else log_loss(Y[valid,j],p,labels=[0,1])
            rows.append(dict(task=task,param=param,validation_loss=loss))
            if best is None or loss<best[0]:best=(loss,m)
        models[task]=best[1];B[:,j]=best[1].predict(x) if j<3 else best[1].predict_proba(x)[:,1]
        splitter=KFold(3,shuffle=True,random_state=SEED) if j<3 else StratifiedKFold(3,shuffle=True,random_state=SEED)
        for a,b in splitter.split(fit,Y[fit,j] if j>=3 else None):
            from sklearn.base import clone
            m=clone(best[1]).fit(x.iloc[fit[a]],Y[fit[a],j]);OOF[fit[b],j]=m.predict(x.iloc[fit[b]]) if j<3 else m.predict_proba(x.iloc[fit[b]])[:,1]
        print('CLINICAL',folder.name,task,len(fit),len(valid),flush=True)
    joblib.dump(models,folder/'clinical_models.joblib');pd.DataFrame(rows).to_csv(folder/'clinical_validation.csv',index=False)
    np.savez_compressed(folder/'baseline.npz',Y=Y,B=B,OOF=OOF)
    return Y,B,OOF

def threshold(y,p):
    if len(np.unique(y))<2:return .5
    ts=np.unique(np.r_[.5,np.quantile(p,np.linspace(0,1,101))]);scores=[balanced_accuracy_score(y,p>=v) for v in ts]
    return float(ts[np.argmax(scores)])

def train_net(t,Y,B,OOF,G,P,C,s,kind,folder,source=None,mode='joint'):
    seed_all();folder.mkdir(parents=True,exist_ok=True)
    matched=t.genotype_matched.to_numpy(bool);train=t.split.eq('train').to_numpy()&matched;valid=t.split.eq('valid').to_numpy()&matched
    mask=np.isfinite(Y);mask[~(train|valid)]=False
    scale=np.maximum(np.nanstd((Y-OOF)[train,:3],axis=0),[.01,.01,.001])
    target=Y.copy();target[:,:3]=(Y[:,:3]-B[:,:3])/scale;target[train,:3]=(Y[train,:3]-OOF[train,:3])/scale
    offset=logit(np.clip(B[:,3:],1e-5,1-1e-5));offset[train]=logit(np.clip(OOF[train,3:],1e-5,1-1e-5))
    target[~(train|valid)]=0;offset[~(train|valid)]=0
    arrays=[G,P,C,np.nan_to_num(target),np.nan_to_num(offset)]
    gx,px,cx,ty,off=[torch.tensor(v,dtype=torch.float32,device=DEVICE) for v in arrays]
    mk=torch.tensor(mask,device=DEVICE)
    ids=np.flatnonzero(train&mask.any(1));v=np.flatnonzero(valid&mask.any(1))
    net=Fusion(s,kind,C.shape[1]).to(DEVICE)
    if source is not None:
        net.load_state_dict(source)
        # New ECOPD residual heads start at zero; encoder features retain UKB training.
        for head in net.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        if mode=='frozen':
            for p in net.genome.parameters():p.requires_grad=False
    groups=[dict(params=[p for n,p in net.named_parameters() if p.requires_grad and not n.startswith('genome.')],lr=3e-4)]
    genome_params=[p for n,p in net.named_parameters() if p.requires_grad and n.startswith('genome.')]
    if genome_params:groups.append(dict(params=genome_params,lr=1e-5 if source is not None else 3e-4))
    opt=torch.optim.AdamW(groups,weight_decay=.05)
    def loss(z,ii):
        lossreg=(z[:,:3]-ty[ii,:3])**2
        losscls=nn.functional.binary_cross_entropy_with_logits(off[ii]+z[:,3:],ty[ii,3:],reduction='none')
        loss=torch.cat([lossreg,losscls],1);m=mk[ii];present=m.sum(0)>0
        return ((loss*m).sum(0)/m.sum(0).clamp(min=1))[present].mean()
    net.eval()
    with torch.no_grad():z,_=net(gx[v],px[v],cx[v]);best=loss(z,v).item()
    state=copy.deepcopy({k:v.cpu() for k,v in net.state_dict().items()});pat=0;epochbest=0;history=[]
    for epoch in range(1,61):
        net.train()
        if mode=='frozen':net.genome.eval()
        order=np.random.default_rng(SEED+epoch).permutation(ids);ls=[]
        for start in range(0,len(order),128):
            ii=order[start:start+128];opt.zero_grad();z,_=net(gx[ii],px[ii],cx[ii]);l=loss(z,ii);l.backward();nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step();ls.append(l.item())
        net.eval()
        with torch.no_grad():z,_=net(gx[v],px[v],cx[v]);vl=loss(z,v).item()
        history.append(dict(epoch=epoch,train_loss=np.mean(ls),validation_loss=vl))
        if vl<best-1e-4:best=vl;state=copy.deepcopy({k:v.cpu() for k,v in net.state_dict().items()});pat=0;epochbest=epoch
        else:pat+=1
        print('EPOCH',folder.name,epoch,vl,flush=True)
        if pat>=10:break
    net.load_state_dict(state);net.eval();deltas=[];emb=[]
    with torch.no_grad():
        for start in range(0,len(t),128):
            z,h=net(gx[start:start+128],px[start:start+128],cx[start:start+128]);deltas.append(z.cpu().numpy());emb.append(h.cpu().numpy())
    delta=np.concatenate(deltas);pred=B.copy();selected={}
    for j,task in enumerate(TASKS):
        ok=valid&np.isfinite(Y[:,j]);best=None
        for w in [0,.1,.25,.5,1]:
            p=B[ok,j]+w*delta[ok,j]*scale[j] if j<3 else expit(logit(np.clip(B[ok,j],1e-5,1-1e-5))+w*delta[ok,j])
            score=mean_squared_error(Y[ok,j],p) if j<3 else log_loss(Y[ok,j],p,labels=[0,1])
            if best is None or score<best[0]-1e-10:best=(score,w,p)
        _,w,p=best;pred[:,j]=B[:,j]+w*delta[:,j]*scale[j] if j<3 else expit(logit(np.clip(B[:,j],1e-5,1-1e-5))+w*delta[:,j])
        selected[task]=dict(weight=w,threshold=threshold(Y[ok,j],p) if j>=3 else None)
    torch.save(dict(state_dict=state,kind=kind,config=CFG,features=FEATURES,tasks=TASKS,scale=scale,selection=selected,best_epoch=epochbest),folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False);np.save(folder/'embeddings.npy',np.concatenate(emb))
    (folder/'selection.json').write_text(json.dumps(selected,indent=2))
    return pred,selected,state

def evaluate(t,Y,p,selected,label,stage,rows,predrows):
    okbase=t.split.eq('test').to_numpy()&t.genotype_matched.to_numpy(bool)
    for j,task in enumerate(TASKS):
        ok=okbase&np.isfinite(Y[:,j]);y=Y[ok,j];q=p[ok,j]
        r=dict(stage=stage,model=label,task=task,n=int(ok.sum()))
        if j<3:r.update(Pearson=float(pearsonr(y,q).statistic),Spearman=float(spearmanr(y,q).statistic),R2=r2_score(y,q),RMSE=mean_squared_error(y,q)**.5)
        else:
            th=selected[task]['threshold'];z=q>=th;both=len(np.unique(y))==2
            tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
            r.update(AUC=roc_auc_score(y,q) if both else None,AUPRC=average_precision_score(y,q) if y.sum()>0 else None,ACC=accuracy_score(y,z),Balanced_ACC=balanced_accuracy_score(y,z),threshold=th,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
        rows.append(r)
        predrows.append(pd.DataFrame(dict(eid=t.loc[ok,'eid'],stage=stage,model=label,task=task,observed=y,predicted=q)))

def base_thresholds(t,Y,B):
    return {task:dict(threshold=threshold(Y[ok,j],B[ok,j]) if j>=3 else None) for j,task in enumerate(TASKS) for ok in [t.split.eq('valid').to_numpy()&np.isfinite(Y[:,j])&t.genotype_matched.to_numpy(bool)]}

def main():
    t,x,g,s=prepare_ukb();train=t.split.eq('train').to_numpy()&t.genotype_matched.to_numpy(bool)
    mu=np.nanmean(g[train],0);sd=np.nanstd(g[train],0);mu=np.nan_to_num(mu);sd=np.where(sd>.01,sd,1)
    gf=np.where(np.isnan(g),mu,g);G=((gf-mu)/sd).astype('float32')
    weights=np.column_stack([s.BETA.to_numpy()*(s['p.value'].to_numpy()<=cut) for cut in [1e-5,1e-4,1e-3,.01]])
    prs=gf@weights;pm=prs[train].mean(0);ps=prs[train].std(0);ps=np.where(ps>1e-6,ps,1);P=((prs-pm)/ps).astype('float32')
    cp=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]);C=cp.fit_transform(numeric(x)[train]);C=cp.transform(numeric(x)).astype('float32')
    joblib.dump(dict(genome_mean=mu,genome_sd=sd,prs_weights=weights,prs_mean=pm,prs_sd=ps,clinical=cp,features=FEATURES),O/'models/preprocessing.joblib')
    Y,B,OOF=clinical_fit(t,x[FEATURES],O/'models/UKB_clinical');rows=[];pr=[];src={}
    evaluate(t,Y,B,base_thresholds(t,Y,B),'Clinical','UKB',rows,pr)
    for kind in ['Transformer','CNN']:
        pred,sel,state=train_net(t,Y,B,OOF,G,P,C,s,kind,O/'models'/('UKB_'+kind));src[kind]=(sel,state)
        evaluate(t,Y,pred,sel,kind,'UKB',rows,pr)
    # Preparation runs independently; fail explicitly if it did not complete.
    e=O/'ECOPD';tt=pd.read_csv(e/'data/participants_targets_splits.csv.gz',dtype={'eid':str});xx=pd.read_csv(e/'data/features.csv.gz',dtype={'eid':str}).set_index('eid').loc[tt.eid].reset_index()
    gg=pd.read_csv(e/'data/genome.csv.gz',dtype={'eid':str}).set_index('eid').loc[tt.eid]
    ge=gg[['G__'+v for v in s.MarkerID]].to_numpy(float);filled=np.where(np.isnan(ge),mu,ge);GE=((filled-mu)/sd).astype('float32');PE=((filled@weights-pm)/ps).astype('float32');CE=cp.transform(numeric(xx)).astype('float32')
    yy=tt[TASKS].to_numpy(float)
    for kind,(sel,state) in src.items():
        net=Fusion(s,kind,C.shape[1]).to(DEVICE);net.load_state_dict(state);net.eval();ds=[]
        with torch.no_grad():
            for start in range(0,len(tt),128):
                args=[torch.tensor(a[start:start+128],device=DEVICE) for a in [GE,PE,CE]];z,_=net(*args);ds.append(z.cpu().numpy())
        delta=np.concatenate(ds);m=joblib.load(O/'models/UKB_clinical/clinical_models.joblib');base=np.column_stack([m[task].predict(xx[FEATURES]) if j<3 else m[task].predict_proba(xx[FEATURES])[:,1] for j,task in enumerate(TASKS)])
        ck=torch.load(O/'models'/('UKB_'+kind)/'model.pt',map_location='cpu',weights_only=False);pp=base.copy()
        for j,task in enumerate(TASKS):
            w=sel[task]['weight'];pp[:,j]=base[:,j]+w*delta[:,j]*ck['scale'][j] if j<3 else expit(logit(np.clip(base[:,j],1e-5,1-1e-5))+w*delta[:,j])
        evaluate(tt,yy,pp,sel,kind,'ECOPD_direct_test',rows,pr)
    YE,BE,OE=clinical_fit(tt,xx[FEATURES],O/'models/ECOPD_clinical')
    evaluate(tt,YE,BE,base_thresholds(tt,YE,BE),'Clinical','ECOPD_local',rows,pr)
    for kind,(_,state) in src.items():
        for mode in ['frozen','light']:
            pp,sel,_=train_net(tt,YE,BE,OE,GE,PE,CE,s,kind,O/'models'/('ECOPD_'+kind+'_'+mode),state,mode)
            evaluate(tt,YE,pp,sel,kind+'_'+mode,'ECOPD_finetune',rows,pr)
    pd.DataFrame(rows).to_csv(O/'results/six_task_metrics.csv',index=False);pd.concat(pr,ignore_index=True).to_csv(O/'results/heldout_predictions.csv.gz',index=False)
    audit=dict(status='complete',tasks=TASKS,source_population='UKB COPD_BL Healthy',external_population='Guangdong baseline COPD flag == 0',variant_discovery='Fixed 4064 COPD-discovered variants; independent of Healthy UKB outcomes; borrowed annotation panel, not Healthy-specific GWAS',train_local=['clinical fitting','3-fold OOF','UKB genome/PRS/clinical normalization','network coefficients'],selection='validation checkpoint and residual weight; thresholds validation only; direct external thresholds frozen UKB',classification='independent BCE heads; not derived from continuous predictions',external_horizon='nominal year 1',seeds=[SEED],limitations=['single seed pilot','baseline non-COPD is broader than strictly normal spirometry','source panel uses COPD associations; no Healthy GWAS discovery','ECOPD fixed legacy clinical mappings and nominal visit years','no causal interpretation of learned genome features'])
    (O/'audit/COMPLETE.json').write_text(json.dumps(audit,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
