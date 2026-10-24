"""Active CAST: explicit TB/TS/BS, variable chunks, shared contextual utility."""
import argparse
import hashlib
import json
import random
import time
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PACKAGE_ROOT=Path(__file__).resolve().parents[1]
ROOT=PACKAGE_ROOT
DATA=None
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
from .graphs import build_views
from .masking import protocol_views
from .chunks import build_cache
from .model import ChunkBank, ThreeViewChunkUtility, pack_chunks
from .chunk_supervision import build_chunk_contexts
from .negatives import build_negative_sets
from .search import search_bundle_beam
from .losses import directional_content_loss
from .batched_search import search_many_pair_beam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--trial',default='cast-run')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--risk-cost-lambda', type=float, default=0.01)
    parser.add_argument('--semantic-base', choices=['recall','structural'], default='recall')
    parser.add_argument('--cost-mode',choices=['train_scaled','raw','saturating'],default='saturating')
    parser.add_argument('--risk-mode',choices=['max','sum'],default='max')
    parser.add_argument('--beam-width',type=int,default=5)
    parser.add_argument('--recall-k',type=int,default=20)
    parser.add_argument('--drop-negative-family',action='append',default=[],
                        choices=['deletion','replacement','redundancy','joint_or_content_omission'])
    parser.add_argument('--no-content-negatives', action='store_true')
    parser.add_argument('--no-chunk-loss',action='store_true')
    parser.add_argument('--chunk-weight',type=float,default=0.1)
    parser.add_argument('--serial-validation',action='store_true')
    parser.add_argument('--view-score-transform',choices=['linear','tanh'],default='tanh')
    parser.add_argument('--chunk-temperature',type=float,default=1.0)
    parser.add_argument('--chunk-margin',type=float,default=0.1)
    parser.add_argument('--resume-state',type=str)
    parser.add_argument('--no-context-chunk', action='store_true')
    parser.add_argument('--no-relational-mask', action='store_true')
    parser.add_argument('--no-chunk-intervention', action='store_true')
    parser.add_argument('--channel-mask', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Only 64 train / 8 validation tasks, 2 updates; never report as full experiment')
    args = parser.parse_args()
    assert torch.cuda.is_available()
    if args.epochs < 1 or args.patience < 1 or args.chunk_temperature<=0 or args.chunk_weight<0:
        raise ValueError('Invalid training schedule')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed+801)
    global DATA
    DATA = args.data_root.resolve()
    ROOT = args.output_root.resolve()
    out = ROOT/args.trial
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run_config.json').exists():
        raise FileExistsError('Use a new trial name to preserve completed or interrupted runs')
    tasks, skills = read(DATA/'tasks.json'), read(DATA/'skill_embedding_records_with_labels.json')
    split = read(DATA/'split_7_2_1.json')
    assert sorted(split['train_task_indices']+split['val_task_indices']+split['test_task_indices']) == list(range(len(tasks)))
    train = split['train_task_indices'][:64] if args.smoke else split['train_task_indices']
    val = split['val_task_indices'][:8] if args.smoke else split['val_task_indices']
    index = {s['skill_id']:i for i, s in enumerate(skills)}
    gold = [[index[s] for s in t['reference_skill_names']] for t in tasks]
    cache_path=DATA/'bge_chunked.npz'
    cache = np.load(cache_path)
    fingerprint = hashlib.sha256(''.join(t['embedding_text'] for t in tasks).encode()+''.join(s['embedding_text'] for s in skills).encode()).hexdigest()
    assert str(cache['fingerprint']) == fingerprint
    chunk_dir = build_cache(out, DATA, cache_path)
    bank = ChunkBank(chunk_dir, len(skills))
    T = torch.tensor(cache['t'], device='cuda', dtype=torch.float32)
    recall_S = torch.tensor(cache['s'], device='cuda', dtype=torch.float32)
    S = recall_S if args.semantic_base=='recall' else bank.skill_means
    meta = torch.tensor([[s['permission_risk_value'], s['token_cost_value']] for s in skills], device='cuda', dtype=torch.float32)
    assert torch.isfinite(meta).all() and (meta >= 0).all()
    # Keep the historical recall universe fixed for controlled comparisons.
    if args.recall_k < args.beam_width or args.recall_k < 2:
        raise ValueError('recall-k must be at least 2 and no smaller than beam-width')
    recall = (T@recall_S.T).topk(args.recall_k, dim=1).indices.cpu().tolist()
    full_views, graph_info = build_views(T, S, gold, train)
    full_views['semantic_base'] = S
    train_costs=np.array([sum(skills[s]['token_cost_value'] for s in gold[ti]) for ti in train])
    cost_scale=float(np.median(train_costs))
    if cost_scale<=0:
        cost_scale=float(np.median(train_costs[train_costs>0])) if (train_costs>0).any() else 1.0
    model = ThreeViewChunkUtility(S.shape[1],full_views,bank,args.risk_cost_lambda,
        cost_mode=args.cost_mode,cost_scale=cost_scale,risk_mode=args.risk_mode,
        view_score_transform=args.view_score_transform).cuda()
    model.no_context_chunk = args.no_context_chunk
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ctx_train = build_chunk_contexts(tasks, gold, bank, train, args.seed)
    ctx_val = build_chunk_contexts(tasks, gold, bank, val, 17)
    write(out/'chunk_context_train_ledger.json', ctx_train)
    write(out/'chunk_context_val_ledger.json', ctx_val)
    write(out/'run_config.json', dict(args=vars(args), graph=graph_info, train_count=len(train), val_count=len(val),
        fingerprint=fingerprint, max_bundle_size=None, max_chunks_per_skill=None,
        implementation='chunk_balanced_v5',cost=args.cost_mode,cost_scale_train_only=cost_scale,
        train_context_records=len(ctx_train), val_context_records=len(ctx_val),
        search='all_pairs_round_beam',initial_pairs=args.recall_k*(args.recall_k-1)//2,beam_width=args.beam_width,contrastive_temperature=0.1,
        target_isolation='per-update exclude scored tasks and their canonical reference bundles in all three graphs',
        objective='bundle_InfoNCE + 0.5*TS_recovery + 0.5*TB_recovery + chunk_weight*directional_content',
        chunk_margin=args.chunk_margin,chunk_temperature=args.chunk_temperature,chunk_in_negative_family=False,
        view_score_transform=args.view_score_transform,
        validation_scheduling='serial' if args.serial_validation else 'cross_task_batched',
        negative_families=['deletion','replacement','redundancy','joint_or_content_omission'],
        context_label='exact literal-support novelty under controlled document-copy intervention; not execution utility',
        recall=f'cached BGE Top-{args.recall_k}', gpu=torch.cuda.get_device_name(0)))
    
    print('SETUP', dict(train=len(train), val=len(val), ctx_train=len(ctx_train), ctx_val=len(ctx_val),
                        chunk_count=len(bank.manifest), max_bundle_size=None), flush=True)
    documents = [s['skill_md_text'].strip() for s in skills]
    negatives = {ti:[s for s in recall[ti] if s not in gold[ti] and documents[s] not in {documents[g] for g in gold[ti]}] for ti in train}

    def gaps(records):
        tis, bundles, coefficients, owners, overrides = [], [], [], [], []
        for owner, r in enumerate(records):
            for j, (b, coefficient) in enumerate(zip(r['bundles'], r['coefficients'])):
                tis.append(r['task_index']); bundles.append(b); coefficients.append(coefficient); owners.append(owner)
                overrides.append(r.get('overrides', [{}]*len(r['bundles']))[j])
        u = pack_chunks(model, T, S, meta, tis, bundles, overrides=overrides)[0]
        return torch.zeros(len(records), device='cuda').scatter_add(0, torch.tensor(owners, device='cuda'),
                u*torch.tensor(coefficients, device='cuda'))

    def contrastive_loss(records):
        # One positive and grouped negatives, all scored by the SAME TB/TS/BS
        # utility. No separate binary classifier, context-margin loss or ranker.
        tis, bundles, overrides, partitions = [], [], [], []
        for r in records:
            positive_index = len(bundles)
            tis.append(r['task_index']); bundles.append(r['positive']); overrides.append({})
            family_indices = []
            for family in r['families'].values():
                local = []
                for negative in family:
                    local.append(len(bundles)); tis.append(r['task_index'])
                    bundles.append(negative['bundle']); overrides.append(negative.get('overrides',{}))
                family_indices.append(local)
            partitions.append((positive_index,family_indices))
        utility = pack_chunks(model,T,S,meta,tis,bundles,overrides=overrides)[0]/0.1
        losses = []
        for positive_index, family_indices in partitions:
            positive = utility[positive_index]
            losses.append(torch.stack([torch.logsumexp(utility[[positive_index]+group],dim=0)-positive
                                       for group in family_indices]).mean())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def evaluate():
        scorer = model.prepare_scoring(T, S, meta)
        batched_results=(None if args.serial_validation else
            search_many_pair_beam(scorer,val,recall,beam_width=args.beam_width))
        rows = []
        for ti in val:
            pred, info = (batched_results[ti] if batched_results is not None else
                search_bundle_beam(lambda bs:scorer(ti,bs),recall[ti],beam_width=args.beam_width,initialization='all_pairs'))
            a, b = set(pred), set(gold[ti])
            raw_cost = sum(skills[s]['token_cost_value'] for s in pred)
            rows.append(dict(task_index=ti, predicted_skill_indices=pred, f1=2*len(a&b)/(len(a)+len(b)),
                jaccard=len(a&b)/len(a|b), size=len(a), risk=max(skills[s]['permission_risk_value'] for s in pred),
                cost=(raw_cost/cost_scale if args.cost_mode=='train_scaled' else
                      raw_cost if args.cost_mode=='raw' else raw_cost/(1+raw_cost)),raw_cost=raw_cost,**info))
        ctx_gaps = []
        for st in range(0, len(ctx_val), 8):
            ctx_gaps.extend(gaps(ctx_val[st:st+8]).cpu().tolist())
        content_deltas=[]
        for st in range(0,len(ctx_val),8):
            ctis=[];cbundles=[];coverrides=[]
            for r in ctx_val[st:st+8]:
                for j in [0,1]:
                    ctis.append(r['task_index']);cbundles.append(r['bundles'][j]);coverrides.append(r['overrides'][j])
            cu=pack_chunks(model,T,S,meta,ctis,cbundles,overrides=coverrides)[0].reshape(-1,2)
            content_deltas.extend((cu[:,0]-cu[:,1]).cpu().tolist())
        return dict(summary={k:float(np.mean([r[k] for r in rows])) for k in ['f1','jaccard','size','risk','cost','raw_cost','seconds']},
                    chunk_validation=dict(n=len(content_deltas),mean_delta=float(np.mean(content_deltas)) if content_deltas else None,
                        common_scale_loss=float(np.logaddexp(0.,.1-np.array(content_deltas)).mean()) if content_deltas else None,
                        positive_fraction=float(np.mean(np.array(content_deltas)>1e-6)) if content_deltas else None,
                        margin_fraction=float(np.mean(np.array(content_deltas)>=.1)) if content_deltas else None,
                        label='held-out document-support proxy, not execution success'),
                    records=rows, context_proxy=dict(n=len(ctx_gaps), preference_accuracy=float(np.mean(np.array(ctx_gaps)>1e-6)) if ctx_gaps else None,
                    numerical_tie_rate=float(np.mean(np.abs(np.array(ctx_gaps))<=1e-6)) if ctx_gaps else None,
                    mean_gap=float(np.mean(ctx_gaps)) if ctx_gaps else None, claim='static literal-coverage proxy, not execution preference'))

    best, bad, history, best_epoch = -1., 0, [], 0
    initial_epoch=0
    if args.resume_state:
        resume_path=Path(args.resume_state)
        state=torch.load(resume_path,weights_only=False,map_location='cpu')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        initial_epoch=state['epoch'];history=state['history'];best=state['best'];bad=state['bad'];best_epoch=state['best_epoch']
        random.setstate(state['python_rng']);rng.setstate(state['sample_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        previous_config=read(resume_path.parent/'run_config.json')['args']
        for key in ['seed','lr','risk_cost_lambda','cost_mode','risk_mode','semantic_base','beam_width',
                    'chunk_weight','chunk_margin','chunk_temperature','view_score_transform','no_chunk_loss']:
            if previous_config[key]!=vars(args)[key]:raise ValueError(f'Resume configuration mismatch: {key}')
        for name in ['best.pt','best_validation.json']:
            shutil.copy2(resume_path.parent/name,out/name)
        print('RESUMED',dict(completed_epochs=initial_epoch,best=best),flush=True)
    start = time.perf_counter()
    for epoch in range(initial_epoch,args.epochs):
        epoch_start=time.perf_counter()
        mode = 'cycle' if args.channel_mask else ('skill' if epoch%2 == 0 else 'bundle')
        if args.no_relational_mask:
            mode = 'none'
        model.views, mask_info = protocol_views(T, S, gold, train, epoch, args.seed, 0.25, mode)
        epoch_channel_keep=model.views.get('channel_keep')
        epoch_channel_probability=model.views.get('channel_probability',0.0)
        write(out/f'epoch_{epoch+1}_mask.json', mask_info)
        graph_done=time.perf_counter()
        model.train()
        # Content omissions have their own directional objective, never InfoNCE negatives.
        records = build_negative_sets(tasks,skills,gold,recall,train,[],args.seed+epoch)
        for r in records:
            r['families']={kind:value for kind,value in r['families'].items() if kind not in args.drop_negative_family}
        records=[r for r in records if r['families']]
        negatives_done=time.perf_counter()
        if epoch == 0:
            write(out/'negative_ledger.json',records)
            write(out/'negative_summary.json',dict(tasks=len(records),
                families={kind:sum(len(r['families'].get(kind,[])) for r in records) for kind in
                          ['deletion','replacement','redundancy','joint_or_content_omission']},
                content_variants=sum(bool(n.get('overrides')) for r in records for group in r['families'].values() for n in group)))
        if args.no_relational_mask:
            members = []
        elif mask_info['mask_type'] == 'skill':
            members = mask_info['ts_recovery_edges']
        elif mask_info['mask_type'] == 'bundle':
            members = [(ti, s) for ti in mask_info['masked_task_indices'] for s in gold[ti]]
        else:
            members = [(ti, s) for ti in train for s in gold[ti]]
        ts = [dict(task_index=ti,positive=[s],families={'hidden_ts':[dict(bundle=[rng.choice(negatives[ti])])]}) for ti,s in members if negatives[ti]]
        tb = ([] if args.no_relational_mask else
              [dict(task_index=ti,positive=gold[ti],families={'hidden_tb':[dict(bundle=gold[ti][:-1]+[rng.choice(negatives[ti])])]})
               for ti in mask_info['masked_task_indices'] if negatives[ti]])
        losses, bundle_losses,ts_losses,tb_losses,chunk_losses = [], [], [], [], []
        content_deltas=[]
        isolation_seconds=0.0
        steps = 2 if args.smoke else (len(train)+31)//32
        for step in range(steps):
            main_batch=rng.sample(records,min(12,len(records)))
            ts_batch=rng.sample(ts,min(16,len(ts))) if ts else []
            tb_batch=rng.sample(tb,min(16,len(tb))) if tb else []
            # Same sampling and target isolation for no-Chunk controls.
            chunk_batch=rng.sample(ctx_train,min(8,len(ctx_train))) if ctx_train else []
            chunk_enabled=(not args.no_chunk_intervention and not args.no_content_negatives)
            # All scored tasks are cold-start for this update, including skill-side messages.
            isolation_start=time.perf_counter()
            query_tasks={r['task_index'] for r in main_batch+ts_batch+tb_batch+chunk_batch}
            hidden_bundles={tuple(b) for b in mask_info['masked_bundle_ids']}
            hidden_bundles.update(tuple(sorted(gold[ti])) for ti in query_tasks)
            context_train=[ti for ti in train if tuple(sorted(gold[ti])) not in hidden_bundles]
            model.views,_=build_views(T,S,gold,context_train,
                masked_bundles=hidden_bundles,masked_skills=mask_info['masked_skill_ids'])
            model.views['semantic_base']=S
            if epoch_channel_keep is not None:
                model.views.update(channel_keep=epoch_channel_keep,channel_probability=epoch_channel_probability)
            assert not model.views['observed'][list(query_tasks)].any()
            if epoch==0 and step==0:
                write(out/'target_isolation_first_update.json',dict(query_tasks=sorted(query_tasks),
                    context_training_tasks=context_train,hidden_bundles=[list(b) for b in sorted(hidden_bundles)],
                    all_queries_isolated=True,masked_skills=mask_info['masked_skill_ids']))
            isolation_seconds+=time.perf_counter()-isolation_start
            bundle_loss=contrastive_loss(main_batch)
            ts_loss=contrastive_loss(ts_batch) if ts_batch else T.new_zeros(())
            tb_loss=contrastive_loss(tb_batch) if tb_batch else T.new_zeros(())
            chunk_loss=T.new_zeros(())
            if chunk_batch and chunk_enabled:
                chunk_tis=[];chunk_bundles=[];chunk_overrides=[]
                for r in chunk_batch:
                    for variant in [0,1]:
                        chunk_tis.append(r['task_index']);chunk_bundles.append(r['bundles'][variant])
                        chunk_overrides.append(r['overrides'][variant])
                chunk_u=pack_chunks(model,T,S,meta,chunk_tis,chunk_bundles,overrides=chunk_overrides)[0].reshape(-1,2)
                chunk_delta=chunk_u[:,0]-chunk_u[:,1]
                chunk_loss=directional_content_loss(chunk_u[:,0],chunk_u[:,1],
                    margin=args.chunk_margin,temperature=args.chunk_temperature)
                content_deltas.extend(chunk_delta.detach().cpu().tolist())
            loss=bundle_loss+0.5*ts_loss+0.5*tb_loss+args.chunk_weight*chunk_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite shared-utility objective')
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(loss.item()); bundle_losses.append(bundle_loss.item())
            ts_losses.append(ts_loss.item());tb_losses.append(tb_loss.item());chunk_losses.append(chunk_loss.item())
        model.views = full_views; model.eval()
        updates_done=time.perf_counter()
        result = evaluate(); f1 = result['summary']['f1']
        validation_done=time.perf_counter()
        if f1 > best+1e-4:
            best, bad, best_epoch = f1, 0, epoch+1
            torch.save(model.state_dict(), out/'best.pt')
            write(out/'best_validation.json', result)
        else:
            bad += 1
        item = dict(epoch=epoch+1, loss=float(np.mean(losses)), bundle_loss=float(np.mean(bundle_losses)),
                    ts_loss=float(np.mean(ts_losses)),tb_loss=float(np.mean(tb_losses)),chunk_loss=float(np.mean(chunk_losses)),
                    chunk_loss_enabled=not args.no_chunk_loss and not args.no_content_negatives,content_negatives=False,
                    chunk_validation=result['chunk_validation'],
                    mask=mask_info['mask_type'], val=result['summary'], context_proxy=result['context_proxy'], bad_epochs=bad)
        item['attention_scale']=float(model.log_attention_scale.exp().clamp(.1,100.).detach())
        item['timing_seconds']=dict(graph=graph_done-epoch_start,
            target_isolation=isolation_seconds,
            negative_generation=negatives_done-graph_done,
            updates_and_sampling=updates_done-negatives_done,
            validation=validation_done-updates_done,total=validation_done-epoch_start)
        item['interaction_weight']=float(model.interaction_logit.sigmoid().detach())
        # Factual intervention metrics, independent of loss scaling.
        if content_deltas:
            item['chunk_train']=dict(n=len(content_deltas),mean_delta=float(np.mean(content_deltas)),
                common_scale_loss=float(np.logaddexp(0.,.1-np.array(content_deltas)).mean()),
                positive_fraction=float(np.mean(np.array(content_deltas)>1e-6)),
                margin_fraction=float(np.mean(np.array(content_deltas)>=.1)))
        item['residual_weights']=model.residual_logits.sigmoid().detach().cpu().tolist()
        history.append(item); print(item, flush=True)
        write(out/'training_progress.json', dict(history=history,best_epoch=best_epoch,best_val_f1=best))
        # Keep optimizer and RNG state so future scheduling fixes can resume
        # at an epoch boundary without silently resetting optimization.
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
            epoch=epoch+1,python_rng=random.getstate(),sample_rng=rng.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),history=history,best=best,bad=bad,best_epoch=best_epoch),out/'last_training_state.pt')
        if bad >= args.patience:
            break
    write(out/'training_summary.json',dict(best_epoch=best_epoch, best_val_f1=best, epochs=len(history),
        seconds=time.perf_counter()-start, stopped_early=len(history)<args.epochs, smoke=args.smoke,
        test_evaluated=False, max_bundle_size=None, ctx_train=len(ctx_train)))
    # Recover exact best-epoch predictions; do not present the final overfit
    # epoch as the selected model. Recompute BOTH relevant semantic baselines.
    best_result = read(out/'best_validation.json')
    comparisons = {'CAST_three_view_chunk_beam':best_result['summary']}
    baseline_records = {}
    semantic_scores = T@bank.skill_means.T
    for representation in ['original_recall','structural_chunks']:
        for k in [2,3,4,5]:
            rows = []
            for ti in val:
                pool = recall[ti]
                order = pool if representation=='original_recall' else sorted(pool,key=lambda s:float(semantic_scores[ti,s]),reverse=True)
                pred = order[:k]; a,b = set(pred),set(gold[ti])
                raw_cost=sum(skills[s]['token_cost_value'] for s in pred)
                rows.append(dict(task_index=ti,predicted_skill_indices=pred,f1=2*len(a&b)/(len(a)+len(b)),
                    jaccard=len(a&b)/len(a|b),size=len(a),risk=max(skills[s]['permission_risk_value'] for s in pred),
                    cost=(raw_cost/cost_scale if args.cost_mode=='train_scaled' else
                          raw_cost if args.cost_mode=='raw' else raw_cost/(1+raw_cost)),raw_cost=raw_cost))
            name=f'{representation}_top{k}'
            comparisons[name]={key:float(np.mean([r[key] for r in rows])) for key in ['f1','jaccard','size','risk','cost','raw_cost']}
            baseline_records[name]=rows
    write(out/'comparison_results.json',dict(summary=comparisons,baseline_records=baseline_records,
          val_count=len(val),smoke=args.smoke,evaluation_split='val',test_evaluated=False,
           cost_definition=dict(mode=args.cost_mode,train_only_scale=cost_scale)))
    lines=['# CAST 三视角、Chunk、Beam 对比结果','',
           f"验证任务数：{len(val)}。{'仅调试子集，不能作为完整实验结论。' if args.smoke else '使用全部验证任务。'}",'',
           '| 方法 | F1 | Jaccard | 平均技能数 | Risk | Cost |','|---|---:|---:|---:|---:|---:|']
    for name,r in comparisons.items():
        lines.append(f"| {name} | {r['f1']:.4f} | {r['jaccard']:.4f} | {r['size']:.3f} | {r['risk']:.4f} | {r['cost']:.4f} |")
    lines.extend(['',f'最佳 epoch={best_epoch}；训练 {len(history)} 轮；早停 patience={args.patience}。',
                  'Original recall 为 BGE Top-20；structural chunks 为同一 Top-20 池内全结构分块均值向量排序。',
                  '新模型从头训练。完整190对初始化再逐轮Beam，独立内容方向损失，每更新隔离评分任务参考图关系，无长度5上限。',
                  f'Cost mode={args.cost_mode}，训练集归一化尺度={cost_scale:.4f}；另报 raw_cost 以便跨版本比较。',
                  '反事实标签为参考与文档证据支持的弱偏好，未报告真实执行成功率。'])
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DONE', dict(best_epoch=best_epoch,best_val_f1=best,smoke=args.smoke),flush=True)


if __name__ == '__main__':
    main()
