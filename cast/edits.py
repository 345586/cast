"""Train-only, evidence-filtered edits scored by the same bundle utility.

Lexical witnesses are weak supervision, never execution or causal verification.
"""
import collections
import hashlib
import itertools
import random

from .text_utils import terms


def build_contextual_constraints(tasks, skills, gold, recall, indices, seed=42, max_bundle_size=5):
    rng = random.Random(seed)
    vocab = [terms(s['skill_md_text']) for s in skills]
    hashes = [hashlib.sha256(s['skill_md_text'].strip().encode()).hexdigest() for s in skills]
    constraints = []

    def emit(ti, bundles, coefficients, kind, weight, evidence):
        if any(not b or (max_bundle_size is not None and len(b) > max_bundle_size) or len(set(b)) != len(b) for b in bundles):
            return
        constraints.append(dict(task_index=ti, bundles=bundles, coefficients=coefficients,
                                margin=0.02, weight=weight, kind=kind, evidence=evidence,
                                verification='reference/document grounded weak supervision; no execution verification'))

    for ti in indices:
        g = list(gold[ti])
        task_terms = terms(tasks[ti]['embedding_text'])
        confidence = 0.45 if tasks[ti]['source_dataset'] == 'skt' else 0.2
        # SAME skill a, SAME task, matched contexts: missing versus covered role.
        # Only compare its marginal utilities; do not force redundant additions
        # to have negative utility solely from word overlap.
        members = g[:]
        rng.shuffle(members)
        for a in members:
            base = [s for s in g if s != a]
            base_terms = set().union(*(vocab[s] for s in base))
            witnesses = (task_terms & vocab[a]) - base_terms
            if not witnesses or (max_bundle_size is not None and len(g) >= max_bundle_size):
                continue
            alternatives = [b for b in recall[ti] if b not in g and witnesses <= vocab[b]]
            alternatives.sort(key=lambda b: (hashes[b] != hashes[a], b))
            if not alternatives:
                continue
            b = alternatives[0]
            exact = hashes[a] == hashes[b]
            emit(ti, [base + [a], base, base + [b, a], base + [b]], [1, -1, -1, 1],
                 'context_role_reversal', confidence if exact else confidence * 0.5,
                 dict(skill=a, covering_skill=b, missing_context=base,
                      unique_task_terms=sorted(witnesses), exact_document_equivalence=exact,
                      claim='larger marginal utility in missing-role context; not a forced negative sign'))
            break

        # Two disjoint task-role witnesses must survive conditioning on all
        # remaining members. Joint restoration outranks BOTH partial repairs.
        candidates = list(itertools.combinations(g, 2))
        rng.shuffle(candidates)
        for a, b in candidates:
            base = [s for s in g if s not in (a, b)]
            base_terms = set().union(*(vocab[s] for s in base))
            wa = (task_terms & vocab[a]) - base_terms - vocab[b]
            wb = (task_terms & vocab[b]) - base_terms - vocab[a]
            if not wa or not wb:
                continue
            evidence = dict(role_a=a, role_b=b, context=base,
                            role_a_terms=sorted(wa), role_b_terms=sorted(wb),
                            claim='joint role completion, not superadditive synergy')
            for partial in (base + [a], base + [b]):
                emit(ti, [g, partial], [1, -1], 'joint_repair', confidence, evidence)
            # For two-member references, never score an empty bundle.
            if base:
                for partial in (base + [a], base + [b]):
                    emit(ti, [partial, base], [1, -1], 'partial_repair', confidence * 0.5, evidence)
            break
    return constraints


def summarize(constraints):
    return dict(total=len(constraints), tasks=len({x['task_index'] for x in constraints}),
                by_kind=dict(collections.Counter(x['kind'] for x in constraints)),
                exact_reversals=sum(x['kind'] == 'context_role_reversal' and
                                    x['evidence']['exact_document_equivalence'] for x in constraints))
