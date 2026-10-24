"""Local auditable content interventions; never similarity-derived utility labels.

Automatic labels prove exact task-literal coverage novelty, not execution gains.
Both contexts keep official member IDs, graph inputs, cardinality and metadata.
"""
import collections
import hashlib
import random
import re
from functools import lru_cache


def task_literals(text):
    literals = re.findall(r'`([^`\n]{3,80})`', text)
    literals += re.findall(r'\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b|--[a-z][a-z-]+|\b[\w-]+\.(?:py|json|csv|yaml|toml|js|ts)\b', text)
    return sorted({w.lower() for w in literals if re.fullmatch(r'[A-Za-z0-9_./:-]+', w) and len(w) >= 3})


def literal_hits(text, literals):
    return set(_literal_hits_cached(text,tuple(sorted(set(literals)))))


@lru_cache(maxsize=65536)
def _literal_hits_cached(text, literals):
    lowered = text.lower()
    return frozenset(w for w in literals if re.search(r'(?<![a-z0-9_])'+re.escape(w)+r'(?![a-z0-9_])', lowered))


def build_chunk_contexts(tasks, gold, bank, indices, seed=42):
    rng = random.Random(seed)
    result = []
    for ti in indices:
        g = list(gold[ti])
        if len(g) < 2:
            continue
        literals = task_literals(tasks[ti]['embedding_text'])
        if not literals:
            continue
        chunks = [(s, ci) for s in g for ci in bank.by_skill[s]]
        hits = {ci:literal_hits(bank.manifest[ci]['text'], literals) for _, ci in chunks}
        options = []
        for owner, ci in chunks:
            other_hits = set().union(*(hits[cj] for _, cj in chunks if cj != ci))
            unique = hits[ci]-other_hits
            if unique:
                options.append((owner, ci, unique))
        if not options:
            continue
        owner, ci, unique = rng.choice(options)
        recipient = rng.choice([s for s in g if s != owner])
        without = [cj for cj in bank.by_skill[owner] if cj != ci]
        duplicate = list(bank.by_skill[recipient])+[ci]
        # Four U calls: restore in missing context minus restore in a context
        # already containing an EXACT copy. Do not use task-chunk cosine signs.
        variants = [{}, {owner:without}, {recipient:duplicate}, {owner:without, recipient:duplicate}]
        result.append(dict(task_index=ti, bundles=[g]*4, coefficients=[1, -1, -1, 1],
            overrides=variants, kind='chunk_context',
            evidence=dict(rule='exact task-literal coverage under duplicate-content intervention',
                          skill=owner, recipient=recipient, chunk=ci, task_literals=sorted(unique),
                          chunk_sha256=bank.manifest[ci]['text_sha256'],
                          oracle_delta_missing=len(unique), oracle_delta_covered=0,
                          member_ids_fixed=True, graph_fixed=True, metadata_fixed=True,
                          claim='documented literal-support novelty; not verified functional utility or execution success')))
    return result


def make_uncapped_edits(tasks, skills, gold, recall, indices, seed=42):
    from contextual_edits import build_contextual_constraints
    rng = random.Random(seed)
    records = []
    equivalent = collections.defaultdict(list)
    docs = [s['skill_md_text'].strip() for s in skills]
    for si, document in enumerate(docs):
        equivalent[hashlib.sha256(document.encode()).hexdigest()].append(si)
    for ti in indices:
        g = list(gold[ti])
        reference_docs = {docs[s] for s in g}
        pool = [s for s in recall[ti] if s not in g and docs[s] not in reference_docs]
        for removed in rng.sample(g, min(2, len(g))):
            records.append(dict(task_index=ti, bundles=[g, [s for s in g if s != removed]], coefficients=[1, -1], kind='restoration'))
        if pool:
            for _ in range(min(2, len(g))):
                old, new = rng.choice(g), rng.choice(pool)
                records.append(dict(task_index=ti, bundles=[g, [s for s in g if s != old]+[new]], coefficients=[1, -1],
                                    kind='replacement', evidence='reference-grounded weak preference; unobserved is not proven invalid'))
        aliases = [a for s in g for a in equivalent[hashlib.sha256(docs[s].encode()).hexdigest()] if a not in g]
        if aliases:
            records.append(dict(task_index=ti, bundles=[g, g+[rng.choice(aliases)]], coefficients=[1, -1],
                                kind='duplicate_addition', evidence='exact duplicate skill document; no new documented content'))
    contextual = build_contextual_constraints(tasks, skills, gold, recall, indices, seed, max_bundle_size=None)
    for r in contextual:
        # For reversal, exact document equivalence is stronger than lexical
        # containment. Keep lexical role witnesses for joint reference repairs.
        if r['kind'] == 'context_role_reversal' and not r['evidence']['exact_document_equivalence']:
            continue
        records.append(r)
    return records
