"""Evaluate a trained CAST checkpoint on validation or test data."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from cast.graphs import build_views
from cast.chunks import build_cache
from cast.model import ChunkBank, ThreeViewChunkUtility
from cast.search import search_bundle_beam


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description='Evaluate CAST on a selected split.')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--recall-k',type=int,default=20)
    p.add_argument('--beam-width',type=int,default=5)
    p.add_argument('--cost-mode',choices=['train_scaled','raw','saturating'])
    p.add_argument('--risk-mode',choices=['max','sum'])
    p.add_argument('--view-score-transform',choices=['linear','tanh'])
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for CAST evaluation')
    data=args.data_root.resolve(); output=args.output_root.resolve(); output.mkdir(parents=True,exist_ok=True)
    tasks=read(data/'tasks.json'); skills=read(data/'skill_embedding_records_with_labels.json'); split=read(data/'split_7_2_1.json')
    index={s['skill_id']:i for i,s in enumerate(skills)}; gold=[[index[x] for x in t['reference_skill_names']] for t in tasks]
    cache=np.load(data/'bge_chunked.npz')
    fingerprint=hashlib.sha256(''.join(t['embedding_text'] for t in tasks).encode()+''.join(s['embedding_text'] for s in skills).encode()).hexdigest()
    if str(cache['fingerprint'])!=fingerprint: raise ValueError('Embedding cache fingerprint mismatch')
    train=split['train_task_indices']; indices=split[f'{args.split}_task_indices']
    bank=ChunkBank(build_cache(output,data,data/'bge_chunked.npz'),len(skills))
    T=torch.tensor(cache['t'],device='cuda',dtype=torch.float32); S=torch.tensor(cache['s'],device='cuda',dtype=torch.float32)
    meta=torch.tensor([[s['permission_risk_value'],s['token_cost_value']] for s in skills],device='cuda',dtype=torch.float32)
    if args.recall_k < args.beam_width or args.recall_k < 2:
        raise ValueError('recall-k must be at least 2 and no smaller than beam-width')
    recall=(T@S.T).topk(args.recall_k,dim=1).indices.cpu().tolist()
    views,graph_info=build_views(T,S,gold,train); views['semantic_base']=S
    config_path=args.checkpoint.parent/'run_config.json'
    saved_args=read(config_path).get('args',{}) if config_path.exists() else {}
    cost_mode=args.cost_mode or saved_args.get('cost_mode','saturating')
    risk_mode=args.risk_mode or saved_args.get('risk_mode','max')
    view_transform=args.view_score_transform or saved_args.get('view_score_transform','tanh')
    cost_scale=float(read(config_path).get('cost_scale_train_only',1.0)) if config_path.exists() else 1.0
    model=ThreeViewChunkUtility(S.shape[1],views,bank,0.01,cost_mode=cost_mode,cost_scale=cost_scale,risk_mode=risk_mode,view_score_transform=view_transform).cuda()
    model.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location='cuda')); model.views=views; model.eval()
    scorer=model.prepare_scoring(T,S,meta); rows=[]; started=time.perf_counter()
    for ti in indices:
        pred,info=search_bundle_beam(lambda bs:scorer(ti,bs),recall[ti],beam_width=args.beam_width,initialization='all_pairs')
        a,b=set(pred),set(gold[ti]); inter=len(a&b); raw=sum(skills[s]['token_cost_value'] for s in pred)
        rows.append(dict(task_index=ti,predicted_skill_indices=pred,precision=inter/len(a) if a else 0,
            recall=inter/len(b) if b else 0,f1=2*inter/(len(a)+len(b)) if a or b else 0,
            jaccard=inter/len(a|b) if a|b else 0,size=len(a),risk=max(skills[s]['permission_risk_value'] for s in pred),
            cost=(raw/cost_scale if cost_mode=='train_scaled' else raw if cost_mode=='raw' else raw/(1+raw)),raw_cost=raw,**info))
    summary={k:float(np.mean([r[k] for r in rows])) for k in ['precision','recall','f1','jaccard','size','risk','cost','raw_cost']}
    result=dict(split=args.split,task_count=len(indices),recall_k=args.recall_k,beam_width=args.beam_width,
        initial_pairs=args.recall_k*(args.recall_k-1)//2,cost_mode=cost_mode,cost_scale=cost_scale,risk_mode=risk_mode,
        view_score_transform=view_transform,graph=graph_info,summary=summary,records=rows,
        evaluation_seconds=time.perf_counter()-started)
    path=output/f'{args.split}_metrics.json'; path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
