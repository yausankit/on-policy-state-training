"""R70d: own-chain query vectors of both encoders on all MuSiQue dev questions, with each step's top-5 (history
masked) and the question-only vector. Usage: viz_flow.py <traces_dir> <out_dir>/own70d.npz TF=<adapter> OP=<adapter>"""
import json, sys, os
from pathlib import Path
import numpy as np
BASE = os.environ.get('OPST_ROOT', str(Path(__file__).resolve().parents[1] / 'src')); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core import Encoder, text19, norm
D29 = Path(BASE + '/inputs/design')
corp_rows = json.load(open(D29 / 'corpus.json')); raw = json.load(open(D29 / 'raw.json'))
corp = [norm(c['title']) + '\n' + norm(c['text']) for c in corp_rows]
idx = {}
for i, c in enumerate(corp_rows): idx.setdefault((norm(c['title']), norm(c['text'])), i)
chain = {r['id']: [idx[(norm(r['paragraphs'][d['paragraph_support_idx']]['title']), norm(r['paragraphs'][d['paragraph_support_idx']]['paragraph_text']))] for d in r['question_decomposition']] for r in raw}
question = {r['id']: r['question'] for r in raw}
H = np.load(D29 / 'doc_emb.npy').astype(np.float32)
traces_dir = Path(sys.argv[1]); adapters = dict(a.split('=', 1) for a in sys.argv[3:])
tr = {s: {t['qid']: [t['writes'][str(k)][0] for k in range(4)] for t in json.load(open(traces_dir / f'{s}.json'))} for s in ('TF', 'OP')}
qids = sorted(tr['TF']); res = dict(qids=np.array(qids), chain=np.array([chain[q] + [-1] * (4 - len(chain[q])) for q in qids]))
for n, path in adapters.items():
    enc = Encoder(path)
    Q = enc.encode([text19(question[q], tr[n][q][:k], corp) for q in qids for k in range(4)]).reshape(len(qids), 4, -1)
    top = np.zeros((len(qids), 4, 5), dtype=np.int64)
    for i, q in enumerate(qids):
        for k in range(4):
            s = H @ Q[i, k]; s[tr[n][q][:k]] = -9; top[i, k] = np.argsort(-s)[:5]
    res[f'{n}_q'] = Q.astype(np.float16); res[f'{n}_w'] = np.array([tr[n][q] for q in qids]); res[f'{n}_top5'] = top; enc.release()
    print(n, 'done', flush=True)
np.savez_compressed(sys.argv[2], **res); print('DONE', len(qids), flush=True)
