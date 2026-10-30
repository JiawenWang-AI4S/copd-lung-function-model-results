"""Participant-level LOOCV with fold-local preprocessing and de novo supervised fits.
Fixed, predeclared hyperparameters; no outer labels used for stopping or selection.
Historical feature selection/GWAS remains exploratory and is explicitly audited.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,time,hashlib,argparse,random,copy,gc
import numpy as np,pandas as pd,joblib,torch
from sklearn.base import clone
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import Ridge
from scipy.special import logit,expit
P=argparse.ArgumentParser();P.add_argument('--cohort',choices=['UKB','Guangdong'],required=True);P.add_argument('--source-root',type=Path,required=True,help='Authorized local directory containing the cohort data/models tree');P.add_argument('--limit',type=int);P.add_argument('--start',type=int,default=0);P.add_argument('--device',default='cuda:6');P.add_argument('--epochs',type=int,default=60);P.add_argument('--seeds',default='20261026,20261027');args=P.parse_args()
ROOT=Path(__file__).resolve().parents[1];SRC=args.source_root.expanduser().resolve();O=ROOT/'outputs'/args.cohort
for d in ['folds','models','audit','results','data']:(O/d).mkdir(exist_ok=True,parents=True)
C=SRC/args.cohort;sys.path.insert(0,str(ROOT/'src'));from aligned import AlignedGenome
TASKS=['FEV1_next','FVC_next','ratio_next','interval_rapid_decline','interval_FVC_rapid_decline','interval_ratio_rapid_decline'];seeds=list(map(int,args.seeds.split(',')));torch.set_num_threads(2);device=args.device if torch.cuda.is_available() else 'cpu'
t=pd.read_csv(C/'data/participants_targets_splits.csv.gz',dtype={'eid':str});schema=pd.read_csv(C/'data/genome_variant_schema.csv');Y=t[TASKS].to_numpy(float);isukb=args.cohort=='UKB';cfg=dict(name='small',dim=16,heads=2,layers=1,patch=64,dropout=.35,lr=3e-4,weight_decay=.03)
if isukb:
 X=pd.read_csv(C/'data/fusion_model_features.csv.gz',dtype={'eid':str}).set_index('eid').loc[t.eid].reset_index();saved=joblib.load(C/'models/clinical_models.joblib');templates={task:saved['Clinical_31SNP__'+task] for task in TASKS};clinical_templates={task:saved['Clinical__'+task] for task in TASKS};raw=np.load(C/'data/genome_inputs.npz');G=raw['dosage'].astype(float);Y[~t.interval_years.between(.5,13).to_numpy()]=np.nan;matched=np.ones(len(t),bool);metadata=joblib.load(C/'models/preprocessing.joblib')
else:
 from residual_lib import Fusion,TOKEN_CONFIGS,offset_fit,linear_delta
 X=pd.read_csv(C/'data/features.csv.gz',dtype={'eid':str}).set_index('eid').loc[t.eid].reset_index();templates=joblib.load(C/'models/clinical_baseline.joblib');clinical_templates=templates;metadata=joblib.load(C/'models/input_preprocessing.joblib');G=pd.read_csv(C/'data/genome.csv.gz',dtype={'eid':str}).set_index('eid').loc[t.eid][['G__'+v for v in schema.MarkerID]].to_numpy(float);A=pd.read_csv(C/'data/panel_ALT_dosages.csv.gz',dtype={'eid':str}).set_index('eid').loc[t.eid][['ALT__'+v for v in metadata['panel_variants']]].to_numpy(float);matched=t.genotype_matched.to_numpy(bool)
elig=np.isfinite(Y);outer=np.flatnonzero(elig.any(1));pd.DataFrame({'eid':t.eid,'eligible_any':elig.any(1),'genotype_matched':matched}).to_csv(O/'data/participants.csv',index=False)
protocol=dict(cohort=args.cohort,primary_tasks=TASKS,outer='Leave one participant out; all records/targets of held participant excluded from clinical fits, OOF fits, train scaling and gradient steps',epochs=args.epochs,seeds=seeds,hyperparameters='Fixed historical clinical estimator settings; small genomic encoder; no model selection against outer predictions',classification_threshold=.5,residual_weight=1.,OOF_folds=3,UKB_residual='mean centered and scaled, preserving historical formula',GD_residual='scaled only; no residual mean',GD_source='fixed UKB-only pretrained representations; no GD outcomes in upstream source training',GD_source_mode='frozen; source hidden features cached deterministically',genetic_feature_discovery='fixed historical variant list and GWAS weights; UKB discovery overlap persists',test_seen_historically=True,interpretation='exploratory LOOCV conditional on previously selected architecture/panels; not nested feature discovery validation',model_scope='UKB Clinical + Clinical_31SNP + Transformer/CNN; GD Clinical_GD + linear SNP/PGI/Extra/All + Token_Context/SNP/PGI/Extra + Transformer/CNN Genetics/All',source_root=str(SRC),device=device)
pfile=O/'audit/protocol.json'
if pfile.exists():
 old=json.loads(pfile.read_text());assert all(old[k]==protocol[k] for k in ['epochs','seeds','primary_tasks']), 'Protocol changed; use a new experiment directory'
else:pfile.write_text(json.dumps(protocol,indent=2))

def fitclinical(ids,template):
 B=np.zeros((len(t),6));models={}
 for j,task in enumerate(TASKS):
  ii=ids[elig[ids,j]];b=template[task];model=clone(b['model']).fit(X.loc[ii,b['features']],Y[ii,j]);B[:,j]=model.predict(X[b['features']]) if j<3 else model.predict_proba(X[b['features']])[:,1];models[task]={**b,'model':model}
 return B,models

def oofclinical(ids):
 Q=np.full((len(t),6),np.nan)
 for j,task in enumerate(TASKS):
  ii=ids[elig[ids,j]];kf=KFold(3,shuffle=True,random_state=20261026) if j<3 else StratifiedKFold(3,shuffle=True,random_state=20261026)
  for aa,bb in kf.split(ii,Y[ii,j] if j>=3 else None):
   fit,hold=ii[aa],ii[bb];b=templates[task];net=clone(b['model']).fit(X.loc[fit,b['features']],Y[fit,j]);Q[hold,j]=net.predict(X.loc[hold,b['features']]) if j<3 else net.predict_proba(X.loc[hold,b['features']])[:,1]
 return Q

def trainnet(factory,inputs,target,offset,mask,seed,lr,wd):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
 net=factory().to(device);opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr,weight_decay=wd);ids=np.flatnonzero(mask.any(1));nt=target.shape[1];isreg=nt==3 and offset is None
 inp={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in inputs.items()};ty=torch.tensor(np.nan_to_num(target),dtype=torch.float32,device=device);mm=torch.tensor(mask,device=device);off=None if offset is None else torch.tensor(np.nan_to_num(offset),dtype=torch.float32,device=device)
 ent=torch.ones(nt,device=device)
 if not isreg:
  q=np.array([target[mask[:,j],j].mean() for j in range(nt)]);ent=torch.tensor(np.maximum(.05,-q*np.log(np.maximum(q,1e-8))-(1-q)*np.log(np.maximum(1-q,1e-8))),dtype=torch.float32,device=device)
 losses=[]
 for epoch in range(args.epochs):
  net.train();order=np.random.default_rng(seed+epoch).permutation(ids);ls=[]
  for start in range(0,len(order),128 if isukb else 64):
   ii=order[start:start+(128 if isukb else 64)];opt.zero_grad();a={k:v[ii] for k,v in inp.items()};out=net(a['G'],a['P'])[0] if isukb else net(a)[0];out=out[:,:nt]
   loss=(out-ty[ii])**2 if isreg else torch.nn.functional.binary_cross_entropy_with_logits(off[ii]+out,ty[ii],reduction='none');m=mm[ii];loss=(((loss*m).sum(0)/m.sum(0).clamp(min=1))/ent);loss=loss[m.sum(0)>0].mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],.5);opt.step();ls.append(float(loss))
  losses.append(float(np.mean(ls)))
 net.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(t),128):
   a={k:v[start:start+128] for k,v in inp.items()};out.append((net(a['G'],a['P'])[0] if isukb else net(a)[0]).cpu().numpy())
 return np.concatenate(out),{k:v.detach().cpu() for k,v in net.state_dict().items()},losses

# GD source is independent of every GD held-out outcome: safe to cache its hidden features.
sourceH={}
if not isukb:
 source=metadata['source'];filled=np.where(np.isnan(G),source['genome_mean'],G);prs=np.column_stack([filled@(schema.BETA.to_numpy()*(schema['p.value'].to_numpy()<=cut)) for cut in [1e-5,1e-4,1e-3,.01]]);baseG=((filled-source['genome_mean'])/source['genome_sd']).astype('float32');baseP=((prs-source['prs_mean'])/source['prs_sd']).astype('float32')
 for kind in ['Transformer','CNN']:
  for group in ['reg','cls']:
   for k,seed in enumerate(seeds):
    net=AlignedGenome(schema,kind,cfg,3 if group=='reg' else 6);ck=torch.load(C/f'models/genome_pretrained/{kind}_{group}_small_{20261023+k}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);net.to(device).eval();hh=[]
    with torch.no_grad():
     for st in range(0,len(t),128):hh.append(net(torch.tensor(baseG[st:st+128],device=device),torch.tensor(baseP[st:st+128],device=device))[1].cpu().numpy())
    sourceH[(kind,group,k)]=np.concatenate(hh);del net

order=outer[args.start:args.start+args.limit if args.limit is not None else None]
for step,hold in enumerate(order):
 fold=O/'folds'/f'{hold:05d}.csv'
 if fold.exists():continue
 begin=time.time();ids=np.array([i for i in range(len(t)) if i!=hold]);assert hold not in ids
 B,base_models=fitclinical(ids,templates);Q=oofclinical(ids);pred={'Clinical_31SNP' if isukb else 'Clinical_GD':B[hold].copy()};stats={};models={};masked=elig.copy();masked[hold]=False;trainmask=masked.copy()
 if isukb:
  plain,_=fitclinical(ids,clinical_templates);pred['Clinical']=plain[hold].copy();mu=np.nanmean(G[ids],axis=0);sd=np.nanstd(G[ids],axis=0);mu=np.where(np.isfinite(mu),mu,metadata['genome_mean']);sd=np.where(np.isfinite(sd)&(sd>1e-6),sd,1);filled=np.where(np.isnan(G),mu,G);prs=np.column_stack([filled@(schema.BETA.to_numpy()*(schema['p.value'].to_numpy()<=cut)) for cut in [1e-5,1e-4,1e-3,.01]]);pm=prs[ids].mean(0);ps=prs[ids].std(0);ps=np.where(ps>1e-6,ps,1.);inputs={'G':((filled-mu)/sd).astype('float32'),'P':((prs-pm)/ps).astype('float32')};stats=dict(genome_mean=mu,genome_sd=sd,prs_mean=pm,prs_sd=ps)
 else:
  genfit=ids[matched[ids]];amu=np.nanmean(A[genfit],axis=0);amu=np.where(np.isfinite(amu),amu,metadata['allele_mean']);qs=np.where(np.isnan(A),amu,A)@metadata['PGI_weights'];raws={'C':X[metadata['clinical_columns']].to_numpy(float),'E':X[metadata['extra_columns']].to_numpy(float),'S':A[:,metadata['snp_indices']],'Q':qs};inputs={};stats={'allele_mean':amu}
  for key,z in raws.items():
   m=np.nan_to_num(np.nanmean(z[genfit],axis=0));filled=np.where(np.isnan(z),m,z);s=filled[genfit].std(0);s=np.where(s>1e-6,s,1);inputs[key]=((filled-m)/s).astype('float32');inputs[key+'M']=np.isnan(z).astype('float32');stats[key+'_mean']=m;stats[key+'_sd']=s
  inputs['QM']=np.column_stack([np.isnan(A[:,abs(metadata['PGI_weights'][:,k])>0]).all(1) for k in range(5)]).astype('float32');inputs.update(G=baseG,P=baseP);trainmask &= matched[:,None]
 residual=Y[:,:3]-Q[:,:3];sdres=np.maximum(np.nanstd(residual[np.where(np.arange(len(t))!=hold)[0]],axis=0),.01);meanres=np.nanmean(residual[ids],axis=0) if isukb else np.zeros(3);targets=(residual-meanres)/sdres;stats.update(residual_sd=sdres,residual_mean=meanres)
 if not isukb:
  for name,keys in {'SNP':['S'],'PGI':['Q'],'Extra':['E'],'All':['E','S','Q']}.items():
   Z=np.concatenate([z for key in keys for z in [inputs[key],inputs[key+'M']]],axis=1);pp=B[hold].copy();entries={}
   for j,task in enumerate(TASKS):
    fit=np.flatnonzero(trainmask[:,j]);model=Ridge(alpha=10).fit(Z[fit],targets[fit,j]) if j<3 else offset_fit(Z[fit],Y[fit,j],logit(np.clip(Q[fit,j],1e-6,1-1e-6)),.1);delta=float(linear_delta(model,Z[[hold]])[0]);pp[j]=B[hold,j]+delta*sdres[j] if j<3 else expit(logit(np.clip(B[hold,j],1e-6,1-1e-6))+delta);entries[task]=model
   pred['Linear_'+name]=pp;models['Linear_'+name]=entries
 neural_specs=['Transformer','CNN'] if isukb else ['Token_Context','Token_SNP','Token_PGI','Token_Extra','Transformer_Genetics','CNN_Genetics','Transformer_All','CNN_All']
 if not isukb and not matched[hold]:neural_specs=['Token_Context','Token_Extra']
 for spec in neural_specs:
  pp=B[hold].copy()
  for group,cols in [('reg',[0,1,2]),('cls',[3,4,5])]:
   outs=[];mask=trainmask[:,cols].copy();target=targets.copy() if group=='reg' else Y[:,3:].copy();offset=None if group=='reg' else logit(np.clip(Q[:,3:],1e-6,1-1e-6));target[hold]=0
   for k,seed in enumerate(seeds):
    local=dict(inputs)
    if isukb:factory=lambda:AlignedGenome(schema,spec,cfg,3)
    else:
     conf=TOKEN_CONFIGS[spec];factory=lambda conf=conf,k=k: Fusion(conf,schema,group,20261023+k)
     if conf['genome']:local['H']=sourceH[(conf['genome'],group,k)]
    out,state,hist=trainnet(factory,local,target,offset,mask,seed,3e-4 if isukb else 5e-4,.03 if isukb else .1);out=out[:,:3];outs.append(out[hold]);models[f'{spec}_{group}_{seed}']=state;stats[f'{spec}_{group}_{seed}_loss_first_last']=[hist[0],hist[-1]]
   delta=np.mean(outs,axis=0)
   pp[cols]=B[hold,cols]+delta*sdres+meanres if group=='reg' else expit(logit(np.clip(B[hold,cols],1e-6,1-1e-6))+delta)
  pred[spec]=pp
 # Unmatched genetic-only results are not evaluable and must not be represented by fake genotypes.
 if not isukb and not matched[hold]:
  for key in ['Linear_SNP','Linear_PGI','Linear_All']:pred.pop(key,None)
 rows=[]
 for spec,pp in pred.items():
  for j,task in enumerate(TASKS):
   if elig[hold,j]:rows.append(dict(eid=t.eid.iloc[hold],held_index=int(hold),spec=spec,task=task,observed=Y[hold,j],predicted=float(pp[j]),predicted_class=int(pp[j]>=.5) if j>=3 else np.nan,threshold=.5 if j>=3 else np.nan,genotype_matched=bool(matched[hold])))
 assert len(rows)>0
 torch.save(dict(models=models,clinical=base_models,fold_preprocessing=stats,held_eid=t.eid.iloc[hold],fit_ids=t.eid.iloc[ids].tolist(),tasks=TASKS,epochs=args.epochs,seeds=seeds),O/'models'/f'{hold:05d}.pt')
 tmp=fold.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(fold)
 (O/'audit'/f'{hold:05d}.json').write_text(json.dumps(dict(held_eid=t.eid.iloc[hold],fit_count=len(ids),held_excluded=True,seconds=time.time()-begin,epochs=args.epochs,seeds=seeds,preprocessing_fitted_without_held=True),indent=2))
 print('FOLD_COMPLETE',args.cohort,int(hold),t.eid.iloc[hold],len(rows),'seconds',round(time.time()-begin,1),flush=True);gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
print('WORKER_COMPLETE',args.cohort,args.start,args.limit,flush=True)
