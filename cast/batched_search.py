"""Same all-pair Beam states, scheduled across tasks at each depth."""
import math
import time
from itertools import combinations


def search_many_pair_beam(score,task_indices,pools,beam_width=5,final_count=10,
                          return_trace=False,tie_tolerance=1e-6):
    if len(task_indices)!=len(set(task_indices)) or not task_indices:
        raise ValueError('Expected distinct nonempty tasks')
    start=time.perf_counter();states={}
    for ti in task_indices:
        pool=list(dict.fromkeys(int(s) for s in pools[ti]))
        if len(pool)<2:raise ValueError('Expected at least two candidates')
        states[ti]=dict(pool=pool,active=[],archive={},trace=[],evaluated=0,
            attempts=0,peak=0,tie_rescores=0)
    for depth in range(2,max(len(s['pool']) for s in states.values())+1):
        requests={};flat_tasks=[];flat_bundles=[]
        for ti,s in states.items():
            if depth>len(s['pool']):continue
            if depth==2:
                children=list(combinations(sorted(s['pool']),2))
            else:
                keys={}
                for bundle,_ in s['active']:
                    for skill in s['pool']:
                        if skill not in bundle:
                            s['attempts']+=1;keys.setdefault(tuple(sorted(bundle+(skill,))),None)
                children=list(keys);s['peak']=max(s['peak'],len(children))
            if not children:continue
            requests[ti]=children;flat_tasks.extend([ti]*len(children));flat_bundles.extend(children)
        if not requests:break
        values=score.many(flat_tasks,flat_bundles)
        if len(values)!=len(flat_bundles) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError('Expected aligned finite scores')
        offset=0
        for ti,children in requests.items():
            s=states[ti];local=list(map(float,values[offset:offset+len(children)]));offset+=len(children)
            ranked=sorted(zip(children,local),key=lambda item:(-item[1],len(item[0]),item[0]))
            # Preserve the serial scorer's decision for close float32 ties.
            top=ranked[:beam_width+1]
            if any(abs(a[1]-b[1])<=tie_tolerance for a,b in zip(top,top[1:])):
                reference=getattr(score,'reference',score)
                ranked=sorted(zip(children,map(float,reference(ti,children))),key=lambda item:(-item[1],len(item[0]),item[0]))
                s['tie_rescores']+=len(children)
            s['evaluated']+=len(children);s['active']=ranked[:beam_width];s['archive'].update(s['active'])
            entry=dict(round=depth,action='all_pairs' if depth==2 else 'expand',
                unique_expansions=len(children),states=[dict(bundle=list(b),utility=u) for b,u in s['active']])
            if depth>2:entry['best_legal_utility']=max(s['archive'].values())
            s['trace'].append(entry)
    elapsed=time.perf_counter()-start;result={}
    for ti,s in states.items():
        ranked=sorted(s['archive'].items(),key=lambda item:(-item[1],len(item[0]),item[0]))
        best_by_length={}
        for item in ranked:best_by_length.setdefault(len(item[0]),item)
        first=sorted(best_by_length.values(),key=lambda item:(-item[1],len(item[0]),item[0]))
        used={b for b,_ in first};final=(first+[item for item in ranked if item[0] not in used])[:final_count]
        best,value=final[0];n=len(s['pool'])
        info=dict(utility=value,evaluated=s['evaluated'],expansion_attempts=s['attempts'],
            initialization='all_pairs',initial_candidates=n*(n-1)//2,
            score_upper_bound=n*(n-1)//2+beam_width*sum(n-k for k in range(2,n)),
            rounds=len(s['trace']),beam_width=beam_width,max_bundle_size=None,
            stop_reason='candidate_pool_exhausted',legal_archive_size=len(s['archive']),
            peak_round_children=s['peak'],tie_verification_scores=s['tie_rescores'],
            final_candidates=[dict(bundle=list(b),utility=u) for b,u in final],
            seconds=elapsed/len(states),batch_wall_seconds=elapsed,
            timing_protocol='amortized multi-task wall time; not isolated serial latency',
            source='batched scheduling of all-pair round Beam')
        if return_trace:info['steps']=s['trace']
        result[ti]=(list(best),info)
    return result
