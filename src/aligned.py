"""Shared genomic encoder and inference helpers for the lung-function models."""
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from scipy.special import expit, logit
from pathlib import Path

TASKS=['FEV1_next','FVC_next','ratio_next','interval_rapid_decline','interval_FVC_rapid_decline','interval_ratio_rapid_decline']
import json

CONFIGS=[dict(name='small',dim=16,heads=2,layers=1,patch=64,dropout=.35,lr=3e-4,weight_decay=.03),dict(name='regularized',dim=32,heads=4,layers=1,patch=32,dropout=.40,lr=1e-4,weight_decay=.05)]
SEEDS=[20261023,20261024]
class AlignedGenome(nn.Module):
 def __init__(self,schema,kind,cfg,ntasks):
  super().__init__();self.kind=kind;self.ntasks=ntasks;d=cfg['dim'];patch=cfg['patch'];n=len(schema);indices=[];chrs=[];coords=[]
  for chrom,rows in schema.groupby('CHR',sort=True):
   ix=rows.index.to_numpy()
   for start in range(0,len(ix),patch):
    part=ix[start:start+patch];indices.append(np.pad(part,(0,patch-len(part)),constant_values=n));chrs.append(int(chrom));bp=schema.iloc[part].POS.mean()/1e8;coords.append([bp,np.sin(bp),np.cos(bp),np.log1p(bp)])
  self.register_buffer('indices',torch.tensor(np.array(indices),dtype=torch.long));self.register_buffer('chromosomes',torch.tensor(chrs,dtype=torch.long));self.register_buffer('coordinates',torch.tensor(coords,dtype=torch.float32))
  self.chr=nn.Embedding(23,d);self.pos=nn.Linear(4,d)
  if kind=='Transformer':
   self.stem=nn.Linear(patch,d);self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,cfg['heads'],d*2,cfg['dropout'],batch_first=True,norm_first=True),cfg['layers'],enable_nested_tensor=False)
  else:
   self.stem=nn.Sequential(nn.Conv1d(1,d,5,padding=2),nn.GELU(),nn.Conv1d(d,d,3,padding=2,dilation=2),nn.GELU(),nn.AdaptiveAvgPool1d(4),nn.Flatten(),nn.Linear(d*4,d));self.enc=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU())
  self.prs=nn.Linear(1,d);self.prs_type=nn.Parameter(torch.randn(4,d)*.02);self.norm=nn.LayerNorm(d);self.query=nn.Parameter(torch.randn(ntasks,d)*.1);self.drop=nn.Dropout(cfg['dropout']);self.heads=nn.ModuleList([nn.Sequential(nn.Linear(d, max(8,d//2)),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(max(8,d//2),1)) for _ in range(ntasks)])
  for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
 def forward(self,x,prs):
  batch=len(x);xp=nn.functional.pad(x,(0,1));patches=xp[:,self.indices]
  if self.kind=='Transformer':tokens=self.stem(patches)
  else:tokens=self.stem(patches.reshape(-1,1,patches.shape[-1])).reshape(batch,len(self.indices),-1)
  tokens=tokens+self.chr(self.chromosomes)[None]+self.pos(self.coordinates)[None];tokens=self.enc(tokens)
  # Four PRS tokens; no direct PRS-to-output shortcut. Learned features still include PRS information.
  pt=self.prs(prs[:,:,None])+self.prs_type[None];tokens=self.norm(torch.cat([tokens,pt],dim=1));scores=torch.einsum('bnd,td->btn',tokens,self.query)/(tokens.shape[-1]**.5);attention=scores.softmax(-1);h=torch.einsum('btn,bnd->btd',attention,tokens);out=torch.cat([head(self.drop(h[:,j])) for j,head in enumerate(self.heads)],dim=1)
  return out,h,attention

def load_inputs(root,features,genome):
 root=Path(root);meta=joblib.load(root/'models/preprocessing.joblib');s=pd.read_csv(root/'data/genome_variant_schema.csv');X=features.copy();gg=genome.copy()
 for df in [X,gg]:
  if 'eid' not in df or df.eid.isna().any() or df.eid.astype(str).duplicated().any():raise ValueError('eid must be unique and nonmissing')
  df['eid']=df.eid.astype(str)
 if set(X.eid)!=set(gg.eid):raise ValueError('Clinical/genome IDs do not match')
 gg=gg.set_index('eid').loc[X.eid];cols=['G__'+v for v in s.MarkerID]
 if not set(cols)<=set(gg):raise ValueError('Missing required genome columns')
 G=gg[cols].apply(pd.to_numeric,errors='raise').to_numpy(float)
 if np.isinf(G).any() or ((G<0)|(G>2)).any():raise ValueError('Genotype dosages must be in [0,2] or missing')
 G=np.where(np.isnan(G),meta['genome_mean'],G);rawprs=np.column_stack([G@(s.BETA.to_numpy()*(s['p.value'].to_numpy()<=cut)) for cut in [1e-5,1e-4,1e-3,.01]])
 for j,c in enumerate(meta['prs_columns']):X[c]=rawprs[:,j]
 G=((G-meta['genome_mean'])/meta['genome_sd']).astype('float32');P=((rawprs-meta['prs_mean'])/meta['prs_sd']).astype('float32')
 return X,G,P,s

def base_predict(root,X,spec='Clinical_31SNP',strict_interval=True):
 models=joblib.load(Path(root)/'models/clinical_models.joblib');out=[]
 for j,task in enumerate(TASKS):
  b=models[spec+'__'+task];missing=set(b['features'])-set(X)
  if missing:raise ValueError('Missing clinical/SNP columns: '+str(sorted(missing)))
  if strict_interval and 'interval_years' in b['features'] and not pd.to_numeric(X.interval_years,errors='coerce').between(.5,13).all():raise ValueError('Planned interval must be in [0.5,13] years')
  out.append(b['model'].predict_proba(X[b['features']])[:,1] if j>=3 else b['model'].predict(X[b['features']]))
 return np.column_stack(out)

def infer_candidate(root,G,P,schema,kind,group,cfg,device='cpu'):
 root=Path(root);outs=[];embs=[]
 for seed in SEEDS:
  ck=torch.load(root/f'models/candidates/{kind}_{group}_{cfg["name"]}_{seed}.pt',map_location=device,weights_only=False);net=AlignedGenome(schema,kind,cfg,3 if group=='reg' else 6).to(device);net.load_state_dict(ck['state_dict']);net.eval();o=[];h=[]
  with torch.no_grad():
   for start in range(0,len(G),128):
    z,e,_=net(torch.tensor(G[start:start+128],device=device),torch.tensor(P[start:start+128],device=device));o.append(z.cpu().numpy());h.append(e.cpu().numpy())
  outs.append(np.concatenate(o));embs.append(np.concatenate(h))
 return np.mean(outs,axis=0),np.concatenate(embs,axis=-1)

def predict(root,features,genome,embedding_dir=None):
 root=Path(root);X,G,P,s=load_inputs(root,features,genome);base=base_predict(root,X);select=json.loads((root/'models/selection.json').read_text());scales=joblib.load(root/'models/residual_scaling.joblib');thresholds=json.loads((root/'models/thresholds.json').read_text());frames=[]
 for kind in ['Transformer','CNN']:
  cache={};out=base.copy()
  for j,task in enumerate(TASKS):
   chosen=select[kind+'__'+task];group='reg' if j<3 else 'cls';cfg=next(c for c in CONFIGS if c['name']==chosen['config']);key=(group,cfg['name'])
   if key not in cache:cache[key]=infer_candidate(root,G,P,s,kind,group,cfg)
   delta,h=cache[key];jj=j if j<3 else j-3;w=chosen['weight']
   if j<3:out[:,j]=base[:,j]+w*delta[:,jj]*scales['sd'][j]+w*scales['mean'][j]
   else:out[:,j]=expit(logit(np.clip(base[:,j],1e-6,1-1e-6))+w*delta[:,jj])
   if embedding_dir:
    path=Path(embedding_dir);path.mkdir(exist_ok=True,parents=True);pd.DataFrame(h[:,jj,:]).assign(eid=X.eid.to_numpy()).to_csv(path/f'{kind}__{task}.csv',index=False)
   th=thresholds[kind+'__'+task];f=pd.DataFrame(dict(eid=X.eid.to_numpy(),spec='Aligned_'+kind,task=task,predicted=out[:,j]));f['threshold']=th if j>=3 else np.nan;f['predicted_class']=(out[:,j]>=th).astype(int) if j>=3 else np.nan;frames.append(f)
 return pd.concat(frames,ignore_index=True)
