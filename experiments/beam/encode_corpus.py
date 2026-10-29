"""Frozen Qwen3-Embedding-0.6B passage embeddings for a long-chain test corpus (same rule as round54 encode_corpus:
raw 'title\\ntext', no prompt, last-token pooling, L2-normalized, fp16). Usage: encode_corpus.py <dataset>"""
import sys, json, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from core import Encoder, norm
ds = sys.argv[1]; d = Path(os.environ.get('LONGCHAIN_DATA', 'data')) / f'{ds}_ref/inputs/test'
corp = [norm(c['title']) + '\n' + norm(c['text']) for c in json.load(open(d / 'corpus.json'))]
enc = Encoder(frozen=True); order = np.argsort([len(c) for c in corp], kind='stable'); out = np.empty((len(corp), 1024), np.float16)
with torch.inference_mode():
    for s in range(0, len(corp), 16):
        ids = order[s:s + 16]
        toks = enc.tokenizer([corp[i] for i in ids], padding=True, truncation=True, max_length=32768, return_tensors='pt').to('cuda')
        h = enc.model(**toks).last_hidden_state[:, -1]
        out[ids] = torch.nn.functional.normalize(h.float(), p=2, dim=1).cpu().numpy().astype(np.float16)
        if s % 64000 == 0: print('encode', s, len(corp), flush=True)
np.save(d / 'doc_emb.npy', out); print('DONE', flush=True)
