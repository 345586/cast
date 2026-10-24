"""Four encoding channels, THREE explicit interaction views, contextual chunks."""
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ChunkBank:
    def __init__(self, folder, skill_count, device='cuda'):
        data = np.load(folder/'vectors.npz')
        self.vectors = torch.tensor(data['embeddings'], device=device, dtype=torch.float32)
        self.manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        self.by_skill = [[] for _ in range(skill_count)]
        for r in self.manifest:
            self.by_skill[r['skill_index']].append(r['chunk_index'])
        if any(not group for group in self.by_skill):
            raise ValueError('Every skill needs at least one content chunk')
        self.skill_means = F.normalize(torch.stack([self.vectors[group].mean(0) for group in self.by_skill]), dim=-1)


def mlp(dim, output, hidden=128):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, output))


class ThreeViewChunkUtility(nn.Module):
    def __init__(self, dim, views, bank, risk_cost_lambda=0.01,
                 cost_mode='train_scaled', cost_scale=1.0, risk_mode='max',view_score_transform='tanh'):
        super().__init__()
        self.views, self.bank = views, bank
        self.risk_cost_lambda = risk_cost_lambda
        self.cost_mode, self.cost_scale, self.risk_mode = cost_mode, float(cost_scale), risk_mode
        if view_score_transform not in ['tanh','linear']:
            raise ValueError('Unknown view score transform')
        self.view_score_transform=view_score_transform
        if self.cost_scale <= 0:
            raise ValueError('Cost normalization uses a positive train-only scale')
        # Cosine attention uses an explicit positive learned scale, not the
        # unnormalized-transformer sqrt(d) rule in either direction.
        self.log_attention_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.interaction_logit = nn.Parameter(torch.tensor(math.log(0.1/0.9)))
        # Independent bounded residuals: semantic, chunk transform, skill->chunk,
        # chunk->skill, and BS combination context. 0.1 is initialization only.
        self.residual_logits = nn.Parameter(torch.full((5,), math.log(0.1/0.9)))
        self.view_enabled = (True,True,True)
        self.semantic_enabled = True
        self.channel_fuse = nn.Sequential(nn.Linear(4*dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.content = mlp(dim, dim)
        self.task_tb_fuse = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.task_ts_fuse = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.graph_strength = nn.Parameter(torch.full((3,), -2.0))
        self.chunk_query = mlp(2*dim, dim)
        self.chunk_content = mlp(dim, dim)
        self.chunk_output = mlp(dim, dim)
        self.bundle_context_message = mlp(dim, dim)
        self.tb_head = mlp(4*dim, 1)
        self.ts_head = mlp(4*dim, 1)
        self.bs_head = mlp(5*dim, 1)
        self.semantic_projection = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.semantic_projection.weight)

    def set_batch(self, task_indices, skill_ids, bundles=None):
        self.task_indices, self.skill_ids, self.bundle_lists = task_indices, skill_ids, bundles

    def train(self, mode=True):
        self._content_cache = None
        return super().train(mode)

    def forward(self, t, s, mask, meta, training_mask=False, chunk_overrides=None, return_details=False):
        batch, width, dim = s.shape
        device = s.device
        ids = self.skill_ids
        tis = torch.as_tensor(self.task_indices, device=device)
        # Overrides alter content only. Graph vectors and skill identities stay fixed.
        raw = s  # Immutable raw-semantic anchor under content interventions.
        content_mean = self.bank.skill_means[ids].clone()
        node_indices, chunk_ids = [], []
        bundle_lists = self.bundle_lists if self.bundle_lists is not None else ids.detach().cpu().tolist()
        for row in range(batch):
            overrides = (chunk_overrides or [{}]*batch)[row]
            for col, si in enumerate(bundle_lists[row]):
                selected = overrides.get(si, self.bank.by_skill[si])
                if selected:
                    node_indices.extend([row*width+col]*len(selected)); chunk_ids.extend(selected)
                    if si in overrides:
                        content_mean[row,col] = F.normalize(self.bank.vectors[selected].mean(0),dim=-1)
                else:
                    content_mean[row,col] = 0
        strength = self.graph_strength.sigmoid()
        residual = self.residual_logits.sigmoid()
        original_base = self.views.get('semantic_base',self.bank.skill_means)[ids]
        cache = getattr(self, '_content_cache', None)
        cached = cache is not None and not self.training and not chunk_overrides and cache['view_id'] == id(self.views)
        graph_ts = F.normalize(original_base+strength[1]*self.views['skill_ts'][ids],dim=-1)
        graph_bs = F.normalize(original_base+strength[2]*self.views['skill_bs'][ids],dim=-1)
        channels = None if cached else torch.stack([raw, self.content(content_mean),
            graph_ts,graph_bs],dim=-2)
        # Channel regularization is distinct from the TB/TS/BS relationship views.
        if not cached and self.training and 'channel_keep' in self.views:
            keep = self.views['channel_keep'][ids].clone()
            keep[..., 0] = True
            channels = channels*keep[..., None]
            channels[..., 1:, :] /= 1-self.views['channel_probability']
        skill_h = (cache['skills'][ids] if cached else F.normalize(self.channel_fuse(channels.flatten(-2)), dim=-1))*mask[..., None]
        count = mask.sum(1).clamp_min(1).to(s.dtype)
        total = (skill_h*mask[..., None]).sum(1)
        # Exclude the queried member: "what does the rest of this bundle provide?"
        other = (total[:, None]-skill_h)/(count-1).clamp_min(1)[:, None, None]
        if getattr(self, 'no_context_chunk', False):
            other = torch.zeros_like(other)
        query = F.normalize(self.chunk_query(torch.cat([t[:, None].expand_as(other), other], -1)), dim=-1)
        flat_q = query.reshape(-1, dim)
        pooled = torch.zeros((batch*width, dim), device=device)
        if chunk_ids:
            owner = torch.tensor(node_indices, device=device)
            ci = torch.tensor(chunk_ids, device=device)
            base_chunk = self.bank.vectors[ci]
            chunk_h = cache['chunks'][ci] if cached else F.normalize(base_chunk+residual[1]*self.chunk_content(base_chunk)+residual[2]*skill_h.reshape(-1,dim)[owner],dim=-1)
            attention_scale = self.log_attention_scale.exp().clamp(0.1,100.0)
            attention_logits = (flat_q[owner]*chunk_h).sum(-1)*attention_scale
            maximum = torch.full((batch*width,), -torch.inf, device=device).scatter_reduce(0, owner, attention_logits, reduce='amax', include_self=True)
            exp = (attention_logits-maximum[owner]).exp()
            denominator = torch.zeros(batch*width, device=device).scatter_add(0, owner, exp)
            attention = exp/denominator[owner].clamp_min(1e-12)
            pooled = pooled.index_add(0, owner, chunk_h*attention[:, None])
        else:
            attention = t.new_empty(0)
        chunk_skill = pooled.reshape(batch, width, dim)
        member_h = F.normalize(skill_h+residual[3]*self.chunk_output(chunk_skill),dim=-1)*mask[..., None]
        # TB candidate encoder is EXACTLY masked mean pooling; no bundle MLP.
        bundle_h = (member_h*mask[..., None]).sum(1)/count[:, None]
        if cached:
            task_tb,task_ts=cache['task_tb'][tis],cache['task_ts'][tis]
        else:
            observed = self.views['observed'][tis, None]
            task_tb = torch.where(observed, F.normalize(t+strength[0]*self.views['task_tb'][tis], dim=-1), t)
            task_ts = torch.where(observed, F.normalize(t+strength[1]*self.views['task_ts'][tis], dim=-1), t)
            task_tb = F.normalize(self.task_tb_fuse(torch.cat([t, task_tb], -1)), dim=-1)
            task_ts = F.normalize(self.task_ts_fuse(torch.cat([t, task_ts], -1)), dim=-1)
        tb = self.tb_head(torch.cat([task_tb, bundle_h, task_tb*bundle_h, (task_tb-bundle_h).abs()], -1)).squeeze(-1)
        tt = task_ts[:, None].expand_as(member_h)
        ts_members = self.ts_head(torch.cat([tt, member_h, tt*member_h, (tt-member_h).abs()], -1)).squeeze(-1)
        ts = (ts_members*mask).sum(1)/count
        # This is explicit combination-context aggregation, NOT a claim of
        # hypergraph incidence/message propagation that the code doesn't define.
        bundle_edge = (chunk_skill*mask[..., None]).sum(1)/count[:, None]
        local_member = F.normalize(chunk_skill+residual[4]*self.bundle_context_message((chunk_skill+bundle_edge[:,None])/2),dim=-1)
        rest = ((member_h*mask[..., None]).sum(1)[:, None]-member_h)/(count-1).clamp_min(1)[:, None, None]
        task = t[:, None].expand_as(member_h)
        bs_members = self.bs_head(torch.cat([local_member, rest, task, local_member*rest, (local_member-rest).abs()], -1)).squeeze(-1)
        bs = (bs_members*mask).sum(1)/count
        view_scores = torch.stack([tb, ts, bs], dim=-1)
        tq = cache['task_semantic'][tis] if cached else F.normalize(t+residual[0]*self.semantic_projection(t),dim=-1)
        sq = cache['skill_semantic'][ids] if cached else F.normalize(raw+residual[0]*self.semantic_projection(raw),dim=-1)
        support = ((tq[:, None]*sq).sum(-1)*mask).sum(1)/count
        enabled = t.new_tensor(self.view_enabled)
        # Ranking utilities need not be probabilities. Linear logits preserve
        # content-edit gradients when a view's common score offset grows.
        transformed=view_scores.tanh() if self.view_score_transform=='tanh' else view_scores
        interaction = (transformed*enabled).sum(-1)/3
        quality = float(self.semantic_enabled)*support+self.interaction_logit.sigmoid()*interaction
        risks = meta[...,0].masked_fill(~mask,0)
        risk = risks.max(1).values if self.risk_mode=='max' else risks.sum(1)
        raw_cost = (meta[..., 1]*mask).sum(1)
        if self.cost_mode != 'saturating':
            raise ValueError('The published method uses saturating cost normalization')
        cost = raw_cost/(1+raw_cost)
        utility = quality-self.risk_cost_lambda*(risk+cost)/2
        if return_details:
            return dict(utility=utility, quality=quality, tb=tb, ts=ts, bs=bs,
                        bundle_h=bundle_h, member_h=member_h, chunk_attention=attention,
                        chunk_owner=node_indices, chunk_ids=chunk_ids, risk=risk, cost=cost,
                        raw_semantic=raw,content_mean=content_mean,graph_ts=graph_ts,graph_bs=graph_bs,
                        semantic_support=support,interaction=interaction,raw_cost=raw_cost)
        return utility, member_h, bundle_h

    @torch.no_grad()
    def prepare_scoring(self, T, S, meta, max_batch_size=256, max_chunk_nodes=24000):
        if self.training:
            raise ValueError('Inference cache requires eval mode')
        if max_batch_size < 1 or max_chunk_nodes < 1:
            raise ValueError('Inference batch limits must be positive')
        strength = self.graph_strength.sigmoid()
        residual = self.residual_logits.sigmoid()
        channels = torch.stack([S, self.content(self.bank.skill_means),
            F.normalize(S+strength[1]*self.views['skill_ts'], dim=-1),
            F.normalize(S+strength[2]*self.views['skill_bs'], dim=-1)], dim=-2)
        h = F.normalize(self.channel_fuse(channels.flatten(-2)), dim=-1)
        owners = torch.tensor([r['skill_index'] for r in self.bank.manifest], device=S.device)
        e = self.bank.vectors
        chunk_h = F.normalize(e+residual[1]*self.chunk_content(e)+residual[2]*h[owners],dim=-1)
        observed=self.views['observed'][:,None]
        task_tb=torch.where(observed,F.normalize(T+strength[0]*self.views['task_tb'],dim=-1),T)
        task_ts=torch.where(observed,F.normalize(T+strength[1]*self.views['task_ts'],dim=-1),T)
        self._content_cache = dict(skills=h,chunks=chunk_h,view_id=id(self.views),
            task_tb=F.normalize(self.task_tb_fuse(torch.cat([T,task_tb],-1)),dim=-1),
            task_ts=F.normalize(self.task_ts_fuse(torch.cat([T,task_ts],-1)),dim=-1),
            task_semantic=F.normalize(T+residual[0]*self.semantic_projection(T),dim=-1),
            skill_semantic=F.normalize(S+residual[0]*self.semantic_projection(S),dim=-1))

        def score_many(task_indices, bundles):
            if len(task_indices)!=len(bundles):
                raise ValueError('Expected aligned tasks and bundles')
            values, batch, batch_tasks, nodes = [], [], [], 0
            for ti,bundle in zip(task_indices,bundles):
                size = sum(len(self.bank.by_skill[si]) for si in bundle)
                if batch and (nodes+size > max_chunk_nodes or len(batch) >= max_batch_size):
                    values.extend(pack_chunks(self, T, S, meta, batch_tasks, batch)[0].cpu().tolist())
                    batch, batch_tasks, nodes = [], [], 0
                batch.append(bundle);batch_tasks.append(ti);nodes += size
            if batch:
                values.extend(pack_chunks(self, T, S, meta, batch_tasks, batch)[0].cpu().tolist())
            return values
        def score(ti,bundles):
            return score_many([ti]*len(bundles),bundles)
        score.many=score_many
        return score


def pack_chunks(model, T, S, meta, task_indices, bundles, overrides=None, details=False):
    if any(not b or len(b) != len(set(b)) for b in bundles):
        raise ValueError('Bundles must be nonempty with distinct official skill IDs')
    width = max(map(len, bundles))
    ids = torch.tensor([list(b)+[0]*(width-len(b)) for b in bundles], device=T.device)
    lengths = torch.tensor([len(b) for b in bundles], device=T.device)
    mask = torch.arange(width, device=T.device)[None, :] < lengths[:, None]
    model.set_batch(task_indices, ids, bundles)
    return model(T[task_indices], S[ids], mask, meta[ids], chunk_overrides=overrides, return_details=details)
