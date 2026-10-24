"""Pair-complete variable-length beam search scored by the shared utility."""
import math
import time
from itertools import combinations


def search_bundle_beam(score,pool,beam_width=5,min_size=2,final_count=10,
                            length_diverse=True,return_trace=False,max_size=None,
                            initialization='prefix_singletons'):
    start=time.perf_counter();pool=list(dict.fromkeys(int(s) for s in pool))
    limit=len(pool) if max_size is None else min(len(pool),max_size)
    if limit<min_size or beam_width<1 or final_count<1:
        raise ValueError('Invalid search bounds or candidate count')
    completed,trace={},[]
    evaluated=0;attempts=0;peak_children=0
    def evaluate(bundles):
        nonlocal evaluated
        values=score(bundles)
        if len(values)!=len(bundles) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError('Expected one finite score per canonical bundle')
        evaluated+=len(bundles)
        return sorted(zip(bundles,map(float,values)),key=lambda item:(-item[1],len(item[0]),item[0]))
    if initialization not in ['prefix_singletons','all_pairs']:
        raise ValueError('Unknown initialization')
    seeds=(list(combinations(sorted(pool),2)) if initialization=='all_pairs' else
           [(s,) for s in pool[:beam_width]])
    active=evaluate(seeds)[:beam_width]
    initial_depth=2 if initialization=='all_pairs' else 1
    if initialization=='all_pairs':
        completed.update((b,u) for b,u in active if len(b)>=min_size)
    trace.append(dict(round=initial_depth,action=initialization,unique_expansions=len(seeds),
        states=[dict(bundle=list(b),utility=u) for b,u in active]))
    for depth in range(initial_depth+1,limit+1):
        keys={}
        for bundle,_ in active:
            for skill in pool:
                if skill not in bundle:
                    attempts+=1;keys.setdefault(tuple(sorted(bundle+(skill,))),None)
        if not keys:break
        peak_children=max(peak_children,len(keys))
        active=evaluate(list(keys))[:beam_width]
        completed.update((b,u) for b,u in active if len(b)>=min_size)
        trace.append(dict(round=depth,action='expand',unique_expansions=len(keys),
            states=[dict(bundle=list(b),utility=u) for b,u in active],
            best_legal_utility=max(completed.values()) if completed else None))
    ranked=sorted(completed.items(),key=lambda item:(-item[1],len(item[0]),item[0]))
    if length_diverse:
        best_by_length={}
        for state in ranked:best_by_length.setdefault(len(state[0]),state)
        first=sorted(best_by_length.values(),key=lambda item:(-item[1],len(item[0]),item[0]))
        used={b for b,_ in first}
        final=(first+[state for state in ranked if state[0] not in used])[:final_count]
    else:final=ranked[:final_count]
    best,value=final[0]
    info=dict(utility=value,evaluated=evaluated,expansion_attempts=attempts,
        initialization=initialization,initial_candidates=len(seeds),
        score_upper_bound=len(seeds)+beam_width*sum(len(pool)-k for k in range(initial_depth,limit)),
        rounds=len(trace),beam_width=beam_width,max_bundle_size=max_size,
        stop_reason='candidate_pool_exhausted' if max_size is None else 'explicit_size_bound',
        legal_archive_size=len(completed),peak_round_children=peak_children,
        final_candidates=[dict(bundle=list(b),utility=u) for b,u in final],
        seconds=time.perf_counter()-start,
        adaptations=['shared utility','pair-complete initialization','variable-length search'])
    if return_trace:info['steps']=trace
    return list(best),info
