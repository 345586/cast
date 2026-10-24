"""Variable-length Markdown chunks. No per-skill chunk-count truncation."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def markdown_blocks(text):
    blocks, current, fence = [], [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            current.append(line)
            if marker and marker.group(1)[0] == fence:
                fence = None
                blocks.append(''.join(current)); current = []
            continue
        if marker:
            if current:
                blocks.append(''.join(current)); current = []
            fence = marker.group(1)[0]; current.append(line)
        elif re.match(r'^\s{0,3}#{1,6}\s', line):
            if current:
                blocks.append(''.join(current))
            current = [line]
        elif not line.strip():
            current.append(line)
            if current:
                blocks.append(''.join(current)); current = []
        else:
            current.append(line)
    if current:
        blocks.append(''.join(current))
    return blocks


def split_document(text, tokenizer, token_limit=448):
    pieces = []
    for block in markdown_blocks(text):
        encoded = tokenizer(block, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        if len(offsets) <= token_limit:
            pieces.append(block)
        else:
            # Encoder input limit, not a maximum number of document chunks.
            cuts = [0]+[offsets[i][0] for i in range(token_limit, len(offsets), token_limit)]+[len(block)]
            pieces.extend(block[a:b] for a, b in zip(cuts, cuts[1:]))
    chunks, pending = [], ''
    for piece in pieces:
        if not piece.strip():
            pending += piece
            continue
        if pending.strip() and len(tokenizer.encode(pending+piece, add_special_tokens=False, truncation=False)) > token_limit:
            chunks.append(pending); pending = ''
        pending += piece
    if pending.strip():
        chunks.append(pending)
    return chunks or [text or '[empty document]']


def build_cache(root, data, base_embeddings):
    import torch
    from sentence_transformers import SentenceTransformer
    skills = json.loads((data/'skill_embedding_records_with_labels.json').read_text(encoding='utf-8'))
    base = np.load(base_embeddings)
    signature = hashlib.sha256(''.join(s['skill_md_text'] for s in skills).encode()).hexdigest()
    out = root/'structural_chunk_cache'; out.mkdir(exist_ok=True)
    if (out/'vectors.npz').exists():
        cached = np.load(out/'vectors.npz')
        if str(cached['document_fingerprint']) != signature or str(cached['base_fingerprint']) != str(base['fingerprint']):
            raise ValueError('Stale structural chunk cache')
        return out
    encoder = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda', local_files_only=False)
    encoder.max_seq_length = 512
    texts, manifest, owners = [], [], []
    for si, skill in enumerate(skills):
        for position, chunk in enumerate(split_document(skill['skill_md_text'], encoder.tokenizer)):
            ci = len(texts); texts.append(chunk); owners.append(si)
            manifest.append(dict(chunk_index=ci, skill_index=si, position=position, text=chunk,
                text_sha256=hashlib.sha256(chunk.strip().encode()).hexdigest()))
    vectors = encoder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.savez_compressed(out/'vectors.npz', embeddings=vectors, owners=np.array(owners),
                        document_fingerprint=signature, base_fingerprint=str(base['fingerprint']))
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = np.bincount(owners, minlength=len(skills))
    (out/'audit.json').write_text(json.dumps(dict(skills=len(skills), chunks=len(texts), min_chunks=int(counts.min()),
        max_chunks=int(counts.max()), mean_chunks=float(counts.mean()), chunk_count_limit=None,
        encoder_token_limit=448, base_fingerprint=str(base['fingerprint']), document_fingerprint=signature), indent=2), encoding='utf-8')
    print('STRUCTURAL_CHUNKS', len(texts), 'max_per_skill', int(counts.max()), flush=True)
    del encoder; torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    raise SystemExit('Use the package training or embedding-preparation command.')
