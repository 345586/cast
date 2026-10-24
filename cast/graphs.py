import torch
from torch import nn
from torch.nn import functional as F
import math
from .utility_base import Utility

def build_views(T,S,gold,training,masked_bundles=None,masked_skills=None):
 # All edges originate from training references; every other task is isolated.
 # Masked bundle/skill IDs lose graph interactions but keep semantic features.
 nt,ns=len(T),len(S)
 masked_bundles={tuple(sorted(b)) for b in (masked_bundles or [])}
 masked_skills=set(masked_skills or [])
 bundles=sorted({tuple(sorted(gold[i])) for i in training if tuple(sorted(gold[i])) not in masked_bundles});bi={b:i for i,b in enumerate(bundles)}
 B=torch.stack([S[[s for s in b if s not in masked_skills]].mean(0) if any(s not in masked_skills for s in b) else S.new_zeros(S.shape[1]) for b in bundles]) if bundles else S.new_zeros((0,S.shape[1]));ntb=len(B)
 tb_edges=[(i,nt+bi[tuple(sorted(gold[i]))]) for i in training if tuple(sorted(gold[i])) in bi]
 ts_edges=[(i,nt+s) for i in training for s in gold[i] if s not in masked_skills]
 edges={'tb':tb_edges,'ts':ts_edges,
        'bs':[(bi[b],ntb+s) for b in bundles for s in b if s not in masked_skills]}
 def propagate(x,pairs):
  if not pairs:
   return F.normalize(x,dim=-1),torch.zeros(len(x),dtype=torch.bool,device=x.device)
  pairs=sorted(set(pairs));a=torch.tensor(pairs,device=x.device).T;ij=torch.cat([a,a.flip(0)],1)
  degree=torch.bincount(ij[0],minlength=len(x)).to(x.dtype);v=(degree[ij[0]]*degree[ij[1]]).rsqrt()
  adj=torch.sparse_coo_tensor(ij,v,(len(x),len(x))).coalesce();cur=x;layers=[x]
  for _ in range(2):cur=torch.sparse.mm(adj,cur);layers.append(cur)
  result=sum(layers)/3
  result=torch.where((degree==0)[:,None],x,result)
  return F.normalize(result,dim=-1),degree>0
 tb,tb_active=propagate(torch.cat([T,B]),edges['tb']);ts,ts_active=propagate(torch.cat([T,S]),edges['ts']);bs,bs_active=propagate(torch.cat([B,S]),edges['bs'])
 observed=torch.zeros(nt,dtype=torch.bool,device=T.device);observed[training]=True
 return dict(task_tb=tb[:nt],task_ts=ts[:nt],skill_ts=ts[nt:],skill_bs=bs[ntb:],observed=observed,
             masked_bundles=sorted([list(b) for b in masked_bundles]),bundle_count=ntb),dict(training_tasks=len(training),reference_bundles=ntb,masked_bundles=len(masked_bundles),edges={k:len(set(v)) for k,v in edges.items()},layers=2,aggregation='mean of layers 0,1,2; semantic fallback for isolated nodes',validation_test_edges=0)

class GraphUtility(Utility):
 def __init__(self,dim,views,mode='full'):
  super().__init__(dim,mode);self.views=views;self.graph_strength=nn.Parameter(torch.full((3,),-2.))
  self.graph_projection=nn.ModuleList([nn.Linear(dim,dim,bias=False) for _ in range(3)])
  for p in self.graph_projection:nn.init.eye_(p.weight)
 def set_batch(self,tis,ids):self.tis=tis;self.ids=ids
 def task_repr(self,t):
  if not getattr(self,'use_adapter',True):return F.normalize(t,dim=-1)
  d=t.shape[-1];a=(self.q(t)@self.k(self.p).T/math.sqrt(d)).softmax(-1);p=a@self.p
  return F.normalize(self.norm(t+torch.sigmoid(self.gamma)*self.adapter(torch.cat([t,p,t*p],-1))),dim=-1)
 def member_logits(self,t,ids,task_indices=None):
  tx=self.task_repr(t)
  if task_indices is not None:
   observed=self.views['observed'][task_indices,None]
   tx=torch.where(observed,F.normalize(tx+self.graph_strength.sigmoid()[1]*self.graph_projection[1](self.views['task_ts'][task_indices]),dim=-1),tx)
  s=self.views['semantic_base'][ids];sts=F.normalize(s+self.graph_strength.sigmoid()[1]*self.graph_projection[1](self.views['skill_ts'][ids]),dim=-1)
  tt=tx[:,None,:].expand_as(sts);return 4*(tt*sts).sum(-1)+getattr(self,'head_residual_weight',.2)*self.ts(torch.cat([tt,sts,tt*sts],-1)).squeeze(-1)
 def forward(self,t,s,mask,meta):
  tx=self.task_repr(t)
  v=self.views;obs=v['observed'][self.tis,None];strength=self.graph_strength.sigmoid()
  if self.training and getattr(self,'task_graph_dropout',0)>0:
   if getattr(self,'task_episode_mask',None) is None:
    self.task_episode_mask=torch.rand(v['observed'].shape,device=obs.device)>=self.task_graph_dropout
   obs=obs & self.task_episode_mask[self.tis,None]
  ttb=torch.where(obs,F.normalize(tx+strength[0]*self.graph_projection[0](v['task_tb'][self.tis]),dim=-1),tx)
  tts=torch.where(obs,F.normalize(tx+strength[1]*self.graph_projection[1](v['task_ts'][self.tis]),dim=-1),tx)
  sts=F.normalize(s+strength[1]*self.graph_projection[1](v['skill_ts'][self.ids]),dim=-1)
  sbs=F.normalize(s+strength[2]*self.graph_projection[2](v['skill_bs'][self.ids]),dim=-1)
  n=mask.sum(1).clamp_min(1);z=F.normalize(self.rho((self.phi(sbs)*mask[:,:,None]).sum(1)/n[:,None]),dim=-1)
  if hasattr(self,'set_encoder'):
   z=torch.zeros_like(tx)
   for length in n.unique().tolist():
    selected=n==length
    z[selected]=self.set_encoder(sbs[selected,:int(length)]).squeeze(1)
   z=F.normalize(z,dim=-1)
  if self.mode=='anchored':z=F.normalize((sbs*mask[:,:,None]).sum(1)/n[:,None]+.1*z,dim=-1)
  tb=self.tb(torch.cat([ttb,z,ttb*z,abs(ttb-z)],-1)).sigmoid().squeeze(-1)
  tt=tts[:,None,:].expand_as(sts);q=self.ts(torch.cat([tt,sts,tt*sts],-1)).sigmoid().squeeze(-1)
  if self.mode=='anchored':
   tb=torch.sigmoid(4*(ttb*z).sum(-1)+getattr(self,'head_residual_weight',.2)*torch.logit(tb.clamp(.001,.999)))
   q=torch.sigmoid(4*(tt*sts).sum(-1)+getattr(self,'head_residual_weight',.2)*torch.logit(q.clamp(.001,.999)))
  ts=self.agg(torch.stack([(q*mask).sum(1)/n,q.masked_fill(~mask,-1).max(1).values,q.masked_fill(~mask,2).min(1).values],-1)).sigmoid().squeeze(-1)
  if self.mode=='anchored':ts=.8*(q*mask).sum(1)/n+.2*ts
  ii,jj=torch.triu_indices(s.shape[1],s.shape[1],1,device=s.device)
  if len(ii):
   si,sj=sbs[:,ii],sbs[:,jj];pm=mask[:,ii]&mask[:,jj];pair=self.bs(torch.cat([tx[:,None,:].expand_as(si),(si+sj)/2,si*sj,abs(si-sj),torch.maximum(si,sj)],-1)).sigmoid().squeeze(-1)
   bs=(pair*pm).sum(1)/pm.sum(1).clamp_min(1);bs=torch.where(n==1,torch.full_like(bs,.5),bs);proxy=((s[:,ii]*s[:,jj]).sum(-1).clamp(0,1)*pm).sum(1)/pm.sum(1).clamp_min(1)
  else:bs=torch.full_like(n,.5);proxy=torch.zeros_like(n)
  sup=self.sup(torch.stack([tb,ts],-1)).sigmoid().squeeze(-1);comp=self.comp(torch.stack([bs,torch.log1p(n)/math.log(6),proxy],-1)).sigmoid().squeeze(-1)
  if self.mode=='anchored':sup=.8*(tb+ts)/2+.2*sup
  risk=(meta[:,:,0]*mask).max(1).values;cost=(meta[:,:,1]*mask).sum(1)/2.5;safe=torch.exp(-(torch.stack([risk,cost],-1)*F.softplus(self.safe_weights)).sum(-1))
  components=torch.stack([sup,comp,safe],-1);gate=.05+.85*self.gate(torch.cat([tx,z,sup[:,None],comp[:,None],risk[:,None],cost[:,None],n[:,None]/5],-1)).softmax(-1)
  return (gate*components).sum(-1),components,gate
