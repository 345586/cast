"""Skill -> bundle -> view masking with global ID sampling and audit counts."""
import random
import torch
from .graphs import build_views


def protocol_views(T, S, gold, training, epoch, seed, probability=0.25, mode='cycle'):
    rng = random.Random(seed + epoch)
    kind = ('skill', 'bundle', 'view')[epoch % 3] if mode == 'cycle' else mode
    unique_skills = sorted({s for ti in training for s in gold[ti]})
    unique_bundles = sorted({tuple(sorted(gold[ti])) for ti in training})
    hidden_skills, hidden_bundles = set(), set()
    if kind == 'skill':
        hidden_skills = set(rng.sample(unique_skills, round(len(unique_skills) * probability)))
    if kind == 'bundle':
        hidden_bundles = set(rng.sample(unique_bundles, round(len(unique_bundles) * probability)))
    active = [ti for ti in training if tuple(sorted(gold[ti])) not in hidden_bundles]
    views, info = build_views(T, S, gold, active, masked_skills=hidden_skills, masked_bundles=hidden_bundles)
    keep = torch.ones((len(S), 4), device=S.device, dtype=torch.bool)
    if kind == 'view':
        # One fixed ID/channel mask for the whole epoch, shared by both sides
        # of each preference. Never independently perturb paired utilities.
        generator = torch.Generator(device=S.device).manual_seed(seed + epoch)
        keep[unique_skills] = torch.rand((len(unique_skills), 4), generator=generator, device=S.device) >= probability
    views.update(semantic_base=S, channel_keep=keep, channel_probability=probability if kind == 'view' else 0.0)
    info.update(epoch=epoch+1, mask_type=kind, mask_probability=probability,
                unique_training_skills=len(unique_skills), unique_training_bundles=len(unique_bundles),
                masked_skill_ids=sorted(hidden_skills), masked_bundle_ids=[list(b) for b in sorted(hidden_bundles)],
                masked_task_indices=[ti for ti in training if ti not in active],
                actual_skill_fraction=len(hidden_skills)/max(1, len(unique_skills)),
                actual_bundle_fraction=len(hidden_bundles)/max(1, len(unique_bundles)),
                masked_channel_entries=int((~keep).sum()),
                ts_recovery_edges=[[ti, s] for ti in training for s in gold[ti] if s in hidden_skills])
    return views, info
