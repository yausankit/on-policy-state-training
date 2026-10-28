"""R70 vector-space visualizations on MuSiQue dev (seed-13 encoders TF=RN and OP), no training.
(1) own-chain trajectories on 4-hop questions, (2) mistake attraction of every contaminated TF state,
(3) gradient x input attribution of the target score over the state's tokens, (4) projection of
f_OP(s) - f_TF(s) on h(a*) - h(wrong). Usage: viz70.py <traces_dir> <out_dir> TF=<adapter> OP=<adapter>"""
import json, sys, random, os
from pathlib import Path
import numpy as np, torch
BASE = os.environ.get('R70_BASE', '/home/amax/revision_20260928'); sys.path.insert(0, BASE + '/round29')
from core29 import Encoder, text19, norm, QUERY_PROMPT

D29 = Path(BASE + '/round29/inputs/design')
corp_rows = json.load(open(D29 / 'corpus.json')); raw = json.load(open(D29 / 'raw.json'))
corp = [norm(c['title']) + '\n' + norm(c['text']) for c in corp_rows]
title = [norm(c['title']) for c in corp_rows]
idx = {}
for i, c in enumerate(corp_rows): idx.setdefault((norm(c['title']), norm(c['text'])), i)
def doc(p): return idx[(norm(p['title']), norm(p['paragraph_text']))]
chain = {r['id']: [doc(r['paragraphs'][d['paragraph_support_idx']]) for d in r['question_decomposition']] for r in raw}
question = {r['id']: r['question'] for r in raw}
H = np.load(D29 / 'doc_emb.npy').astype(np.float32)
traces_dir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
adapters = dict(a.split('=', 1) for a in sys.argv[3:])
traces = {s: {t['qid']: [t['writes'][str(k)][0] for k in range(4)] for t in json.load(open(traces_dir / f'{s}.json'))} for s in ('TF', 'OP')}

# states of the TF chains (as in the recovery probe) and both encoders' own chains
states = []
for qid, w in traces['TF'].items():
    g = chain[qid]; hist = []
    for k in range(4):
        miss = [d for d in g if d not in hist]
        if miss: states.append(dict(qid=qid, hist=list(hist), wrong=[d for d in hist if d not in g], target=miss[0], missing=miss))
        hist.append(w[k])
own = {s: [(qid, w[:k]) for qid, w in traces[s].items() if len(chain[qid]) == 4 for k in range(4)] for s in ('TF', 'OP')}

def text(qid, hist): return text19(question[qid], hist, corp)
def spans(S, qid, n):
    q = S.find('Original question: '); e = S.find('Evidence state:'); seg = [('question', q, e)]; pos = e
    starts = []
    for i in range(n):
        pos = S.find(f'[{i + 1}] ', pos); starts.append(pos)
    for i in range(n): seg.append((i, starts[i], starts[i + 1] if i + 1 < n else len(S)))
    return seg

contam = [i for i, s in enumerate(states) if s['wrong']]
random.seed(13); f = {}; ownf = {}; attr = {}; examples = {}
bremen = next((q for q in question if 'Bremen Cathedral' in question[q]), None)
for name, path in adapters.items():
    enc = Encoder(path)
    f[name] = enc.encode([text(s['qid'], s['hist']) for s in states])
    ownf[name] = enc.encode([text(q, h) for q, h in own[name]])
    emb_layer = enc.model.get_input_embeddings(); A = []
    for j, i in enumerate(contam):
        s = states[i]; S = QUERY_PROMPT + norm(text(s['qid'], s['hist']))
        tok = enc.tokenizer(S, return_offsets_mapping=True, return_tensors='pt')
        off = tok.pop('offset_mapping')[0].tolist(); tok = tok.to('cuda')
        e = emb_layer(tok['input_ids']).detach().float().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.float16):
            v = enc.model(inputs_embeds=e, attention_mask=tok['attention_mask']).last_hidden_state[0, -1]
        v = torch.nn.functional.normalize(v.float(), dim=0)
        (v @ torch.as_tensor(H[s['target']], device='cuda')).backward()
        a = (e.grad[0] * e[0]).sum(-1).abs().detach().cpu().numpy()
        mass = dict(question=0., gold=0., wrong=0.); ntok = dict(question=0, gold=0, wrong=0)
        for lab, lo, hi in spans(S, s['qid'], len(s['hist'])):
            key = 'question' if lab == 'question' else ('wrong' if s['hist'][lab] in s['wrong'] else 'gold')
            sel = [t for t, (a0, a1) in enumerate(off) if a0 >= lo and a1 <= hi and a1 > a0]
            mass[key] += float(a[sel].sum()); ntok[key] += len(sel)
        A.append(dict(state=i, mass=mass, ntok=ntok))
        if s['qid'] == bremen or j < 40:
            examples.setdefault(i, {})[name] = dict(text=S, off=off, attr=a.tolist(), spans=spans(S, s['qid'], len(s['hist'])))
        if j % 200 == 0: print(name, 'attr', j, len(contam), flush=True)
    attr[name] = A; enc.release(); torch.cuda.empty_cache()

def cos(a, b): return float(a @ b)
# (2) mistake attraction
att = [dict(state=i, **{n: dict(wrong=max(cos(f[n][i], H[w]) for w in states[i]['wrong']), target=cos(f[n][i], H[states[i]['target']]),
        hit=int(int(np.argmax(np.where(np.isin(np.arange(len(H)), states[i]['hist']), -9, H @ f[n][i]))) in states[i]['missing'])) for n in f}) for i in contam]
# (4) representation difference
proj = []
for i in contam:
    s = states[i]; d = f['OP'][i] - f['TF'][i]; wbar = H[s['wrong']].mean(0)
    sc = H @ f['TF'][i]; sc[s['hist'] + chain[s['qid']]] = -9; hard = int(np.argmax(sc))
    u = H[s['target']] - wbar; un = H[hard] - wbar
    proj.append(dict(state=i, norm=float(np.linalg.norm(d)), target_dir=float(d @ u / np.linalg.norm(u)), hard_dir=float(d @ un / np.linalg.norm(un)),
                     target_minus_hard=float(d @ (H[s['target']] - H[hard]) / np.linalg.norm(H[s['target']] - H[hard]))))
# (1) trajectories: 4-hop questions, PCA over gold, every written passage and both query paths
traj = []
for qid in traces['TF']:
    if len(chain[qid]) != 4: continue
    pts, lab = [], []
    for d in dict.fromkeys(chain[qid] + traces['TF'][qid] + traces['OP'][qid]):
        pts.append(H[d]); lab.append(dict(kind='gold' if d in chain[qid] else 'wrong', doc=d, title=title[d], hop=chain[qid].index(d) + 1 if d in chain[qid] else None,
                                         by=[n for n in ('TF', 'OP') if d in traces[n][qid]]))
    for n in ('TF', 'OP'):
        for k in range(4):
            pts.append(ownf[n][own[n].index((qid, traces[n][qid][:k]))]); lab.append(dict(kind='query', enc=n, step=k + 1, write=traces[n][qid][k]))
    X = np.stack(pts); Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
    ng = lambda n: sum(d in chain[qid] for d in traces[n][qid])
    traj.append(dict(qid=qid, question=question[qid], gold_found=dict(TF=ng('TF'), OP=ng('OP')), var2=float((S[:2] ** 2).sum() / (S ** 2).sum()),
                     points=[dict(xy=P[i].tolist(), **lab[i]) for i in range(len(pts))],
                     cos=[dict(enc=n, step=k + 1, gold=[cos(ownf[n][own[n].index((qid, traces[n][qid][:k]))], H[g]) for g in chain[qid]]) for n in ('TF', 'OP') for k in range(4)]))
json.dump(dict(states=[dict(qid=s['qid'], hist=s['hist'], wrong=s['wrong'], target=s['target']) for s in states], contam=contam, bremen=bremen,
               attraction=att, projection=proj, attribution=attr, trajectories=traj, titles={str(d): title[d] for s in states for d in s['hist'] + [s['target']]}),
          open(out / 'viz70.json', 'w'))
json.dump({str(k): v for k, v in examples.items()}, open(out / 'examples70.json', 'w'))
print('DONE', len(states), len(contam), len(traj), flush=True)
