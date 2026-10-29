"""Query vectors of both encoders along their own chains on MuSiQue, with gold and written passage vectors, for
Figure 4 (trajectories.py). Usage: viz_trajectories.py <traces_dir> <out_dir>/own70c.npz TF=<adapter> OP=<adapter>"""
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
traces_dir = Path(sys.argv[1]); adapters = dict(a.split('=', 1) for a in sys.argv[3:])
tr = {s: {t['qid']: [t['writes'][str(k)][0] for k in range(4)] for t in json.load(open(traces_dir / f'{s}.json'))} for s in ('TF', 'OP')}
qids = sorted(q for q in tr['TF'] if len(chain[q]) >= 3)
res = dict(qids=np.array(qids), chain=np.array([chain[q] + [-1] * (4 - len(chain[q])) for q in qids]))
for n, path in adapters.items():
    enc = Encoder(path)
    res[f'{n}_q'] = enc.encode([text19(question[q], tr[n][q][:k], corp) for q in qids for k in range(4)]).reshape(len(qids), 4, -1)
    res[f'{n}_w'] = np.array([tr[n][q] for q in qids]); enc.release()
np.savez_compressed(sys.argv[2], **res); print('DONE', len(qids), flush=True)
