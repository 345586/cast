"""Four negative families, one reference-anchored contrastive objective."""
import collections
import hashlib
import random


def build_negative_sets(tasks, skills, gold, recall, indices, chunk_contexts, seed=42):
    from .chunk_supervision import task_literals,literal_hits
    rng = random.Random(seed)
    documents = [s['skill_md_text'].strip() for s in skills]
    hashes = [hashlib.sha256(d.encode()).hexdigest() for d in documents]
    aliases = collections.defaultdict(list)
    for si, h in enumerate(hashes):
        aliases[h].append(si)
    contextual = {r['task_index']:r for r in chunk_contexts}
    result = []
    for ti in indices:
        g = list(gold[ti])
        refdocs = {hashes[s] for s in g}
        candidates = [s for s in recall[ti] if s not in g and hashes[s] not in refdocs]
        literals=task_literals(tasks[ti]['embedding_text'])
        member_hits={s:literal_hits(documents[s],literals) for s in g}
        families = {}
        def add(kind, bundle, evidence, overrides=None):
            if bundle and len(bundle)==len(set(bundle)):
                families.setdefault(kind, []).append(dict(bundle=bundle, overrides=overrides or {}, evidence=evidence))
        for old in rng.sample(g, min(2,len(g))):
            add('deletion', [s for s in g if s != old], dict(removed=old, label='reference-member omission; weak utility preference'))
        if candidates:
            for old in rng.sample(g, min(2,len(g))):
                others=set().union(*(member_hits[s] for s in g if s!=old))
                witness=member_hits[old]-others
                # Do not label a substitute negative merely because its ID
                # was unobserved. Require documented reference support lost.
                mismatched=[s for s in candidates if witness and not literal_hits(documents[s],sorted(witness))]
                if not mismatched:
                    continue
                new = rng.choice(mismatched)
                add('replacement', [s for s in g if s != old]+[new],
                    dict(removed=old,inserted=new,lost_task_literals=sorted(witness),
                         label='documented task-literal support lost; functional alternatives remain possible',confidence='document_support_proxy'))
        risk = max(skills[s]['permission_risk_value'] for s in g)
        # A duplicate must introduce a strictly greater explicit burden; a free
        # identical-content alias is a tie, not a fabricated negative.
        duplicates = [a for s in g for a in aliases[hashes[s]] if a not in g and
                      (skills[a]['token_cost_value'] > 0 or skills[a]['permission_risk_value'] > risk)]
        if duplicates:
            a = rng.choice(duplicates)
            add('redundancy', g+[a], dict(added=a, document_sha256=hashes[a],
                label='exact duplicate content and increased risk/cost proxy'))
        if len(g) >= 3:
            removed = rng.sample(g,2)
            add('joint_or_content_omission',[s for s in g if s not in removed],
                dict(removed=removed,label='joint reference omission; not a superadditivity claim'))
        ctx = contextual.get(ti)
        if ctx:
            # The skill ID set stays identical. Drop only the chunk with a
            # proven unique occurrence of an explicit task literal.
            add('joint_or_content_omission',g,ctx['evidence'],ctx['overrides'][1])
        if families:
            result.append(dict(task_index=ti, positive=g, families=families))
    return result
