"""R69 G-Retriever retrieval-only baseline on HippoRAG 2 OpenIE textual graphs.
Usage: gr69.py <musique|hotpot_pool|2wiki_pool>
Calls the official retrieval_via_pcst (G-Retriever repo, unmodified) with WebQSP defaults topk=3, topk_e=5, cost_e=0.5."""
import json, re, sys, time, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.environ.get('WORK', '.') + '/round69/gretriever/G-Retriever')
from src.dataset.utils.retrieval import retrieval_via_pcst
from torch_geometric.data.data import Data
from transformers import AutoTokenizer, AutoModel

W = Path(os.environ.get('WORK', '.') + '/round69/gretriever'); R54 = Path(os.environ.get('WORK', '.') + '/round57/round54/data')
MODEL = os.environ.get('WORK', '.') + '/round8/encoder/model'
QP = 'Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: '
TOPK, TOPK_E, COST_E = 3, 5, 0.5
norm = lambda t: ' '.join(str(t or '').split())
proc = lambda t: re.sub('[^A-Za-z0-9 ]', ' ', str(t).lower()).strip()   # HippoRAG 2 text_processing
key = lambda s: ' '.join(s.split())
read = lambda p: json.load(open(p))

def load(ds):
    if ds == 'musique':
        d = Path(os.environ.get('WORK', '.') + '/round29/inputs/design')
        corp = [norm(c['title']) + '\n' + norm(c['text']) for c in read(d / 'corpus.json')]
        idx = {key(c): i for i, c in enumerate(corp)}
        qs, gold, typ = [], {}, {}
        for r in read(d / 'raw.json'):
            q = str(r['id']); qs.append((q, r['question']))
            gold[q] = sorted({idx[key(norm(p['title']) + '\n' + norm(p['paragraph_text']))] for p in r['paragraphs'] if p['is_supporting']})
            typ[q] = r['id'].split('__')[0][:4]
        return corp, qs, gold, typ, np.load(d / 'doc_emb.npy', mmap_mode='r'), W / 'openie_design.json', None
    if ds == 'hotpot_pool':
        d = R54 / 'hotpotqa_long_pool/inputs/test'; oie = W / 'openie_hotpot_pool.json'; ids = None
    else:
        d = R54 / '2wiki_long_ref/inputs/test'; oie = W / 'openie_2wiki_pool.json'
    full = [norm(c['title']) + '\n' + norm(c['text']) for c in read(d / 'corpus.json')]
    gold = {k: [int(x) for x in v] for k, v in read(d / 'ordered_gold.json').items()}
    raw = {r['_id']: r for r in read(d / 'raw.json')}
    qs = [(q['qid'], q['question']) for q in read(d / 'questions.json')]
    typ = {q: raw[q]['type'] for q, _ in qs}
    E = np.load(d / 'doc_emb.npy', mmap_mode='r')
    if ds == '2wiki_pool':   # rebuild the R63 pool: corpus indices of the OpenIE pool passages, sorted
        idx = {key(c): i for i, c in enumerate(full)}
        ids = sorted({idx[key(x['passage'])] for x in read(oie)['docs']})
        g = {d_ for v in gold.values() for d_ in v}; assert g <= set(ids), 'gold not in pool'
        new = {o: i for i, o in enumerate(ids)}
        full = [full[i] for i in ids]; gold = {k: [new[x] for x in v] for k, v in gold.items()}; E = np.asarray(E[ids])
    return full, qs, gold, typ, E, oie, ids

class Enc:
    def __init__(s):
        s.tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left', local_files_only=True)
        s.m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    @torch.inference_mode()
    def __call__(s, texts, bs=64):
        order = np.argsort([len(t) for t in texts], kind='stable'); out = np.empty((len(texts), 1024), np.float32)
        for a in range(0, len(texts), bs):
            ii = order[a:a + bs]
            t = s.tok([texts[i] for i in ii], padding=True, truncation=True, max_length=32768, return_tensors='pt').to('cuda')
            h = s.m(**t).last_hidden_state[:, -1]
            out[ii] = torch.nn.functional.normalize(h.float(), p=2, dim=1).cpu().numpy()
        return out

class Rec:  # stands in for the textual node/edge DataFrames; records which rows the official function selects
    def __init__(s, n): s.n = n; s.sel = None
    def __len__(s): return s.n
    @property
    def iloc(s): return s
    def __getitem__(s, i): s.sel = np.asarray(i, dtype=np.int64).reshape(-1); return s
    def to_csv(s, **kw): return ''

def metrics(rk, gold, typ, ks=(5, 10)):
    per, allm = {}, []
    for q, r in rk.items():
        g = gold[q]; m = {}
        for k in ks:
            hit = len(set(g) & set(r[:k])); m[f'R@{k}'] = 100 * hit / len(g); m[f'C@{k}'] = 100.0 * (hit == len(g))
        allm.append(m); per.setdefault(typ[q], []).append(m)
    avg = lambda rows: {k: sum(x[k] for x in rows) / len(rows) for k in rows[0]}
    return dict(overall=avg(allm), n=len(allm), by_type={t: dict(n=len(v), **avg(v)) for t, v in sorted(per.items())})

def main(ds):
    t0 = time.time(); out = W / ds; out.mkdir(exist_ok=True)
    corp, qs, gold, typ, DE, oie, ids = load(ds)
    pidx = {key(c): i for i, c in enumerate(corp)}
    docs = read(oie)['docs']; unmatched = 0
    node_id, node_txt, node_p = {}, [], []
    pred_id, pred_txt = {}, []
    edge_id, e_src, e_dst, e_pred, edge_p = {}, [], [], [], []
    def nid(t):
        if t not in node_id: node_id[t] = len(node_txt); node_txt.append(t); node_p.append(set())
        return node_id[t]
    n_self = 0
    for x in docs:
        p = pidx.get(key(x['passage']))
        if p is None: unmatched += 1; continue
        for e in x.get('extracted_entities') or []:
            t = proc(e)
            if t: node_p[nid(t)].add(p)
        for tr in x.get('extracted_triples') or []:
            if not isinstance(tr, list) or len(tr) != 3: continue
            s, r, o = proc(tr[0]), norm(tr[1]), proc(tr[2])
            if not s or not o or not r: continue
            if s == o: n_self += 1; continue
            a, b = nid(s), nid(o); node_p[a].add(p); node_p[b].add(p)
            if r not in pred_id: pred_id[r] = len(pred_txt); pred_txt.append(r)
            k = (a, pred_id[r], b)
            if k not in edge_id: edge_id[k] = len(e_src); e_src.append(a); e_dst.append(b); e_pred.append(pred_id[r]); edge_p.append(set())
            edge_p[edge_id[k]].add(p)
    stats = dict(dataset=ds, corpus=len(corp), openie_docs=len(docs), unmatched_docs=unmatched, nodes=len(node_txt), edges=len(e_src),
                 predicates=len(pred_txt), dropped_self_loops=n_self, passages_with_graph=len(set().union(*node_p)))
    print(stats, flush=True)
    enc = Enc()
    NX = torch.from_numpy(enc(node_txt)); PX = enc(pred_txt)
    Q = torch.from_numpy(enc([QP + norm(q) for _, q in qs], bs=32))
    del enc; torch.cuda.empty_cache()
    EX = torch.from_numpy(PX[np.asarray(e_pred)]); del PX
    graph = Data(x=NX, edge_index=torch.LongTensor([e_src, e_dst]), edge_attr=EX, num_nodes=len(node_txt))
    DE = torch.from_numpy(np.asarray(DE, dtype=np.float32))
    print('encoded', time.time() - t0, flush=True)
    rankings, info = {}, {}
    fb5 = fb10 = 0
    for qi, (q, _) in enumerate(qs):
        qe = Q[qi:qi + 1]
        tn, te = Rec(graph.num_nodes), Rec(graph.num_edges)
        retrieval_via_pcst(graph, qe, tn, te, topk=TOPK, topk_e=TOPK_E, cost_e=COST_E)
        sn, se = tn.sel, te.sel
        ns = (NX[sn] @ qe[0]).tolist() if len(sn) else []; es = (EX[se] @ qe[0]).tolist() if len(se) else []
        best = {}
        for v, sc in zip(sn.tolist(), ns):
            for p in node_p[v]: best[p] = max(best.get(p, -9), sc)
        for e, sc in zip(se.tolist(), es):
            for p in edge_p[e]: best[p] = max(best.get(p, -9), sc)
        dense = (DE @ qe[0]).numpy()
        graph_rank = sorted(best, key=lambda p: (-best[p], -dense[p], p))
        seen = set(graph_rank); rk = graph_rank[:50]
        if len(rk) < 50:
            for p in np.argsort(-dense, kind='stable')[:50 + len(seen)]:
                if int(p) not in seen: rk.append(int(p))
                if len(rk) >= 50: break
        rankings[q] = rk[:10]
        info[q] = dict(sel_nodes=len(sn), sel_edges=len(se), graph_passages=len(graph_rank))
        fb5 += max(0, 5 - len(graph_rank)); fb10 += max(0, 10 - len(graph_rank))
        if qi % 100 == 0: print(qi, len(sn), len(se), len(graph_rank), round(time.time() - t0), flush=True)
    m = metrics(rankings, gold, typ)
    gp = [v['graph_passages'] for v in info.values()]
    stats.update(questions=len(qs), fallback_ranks_top5=fb5, fallback_ranks_top10=fb10, total_ranks_top10=10 * len(qs),
                 questions_with_any_fallback_top10=sum(g < 10 for g in gp), mean_graph_passages=float(np.mean(gp)),
                 median_graph_passages=float(np.median(gp)), mean_sel_nodes=float(np.mean([v['sel_nodes'] for v in info.values()])),
                 mean_sel_edges=float(np.mean([v['sel_edges'] for v in info.values()])), settings=dict(topk=TOPK, topk_e=TOPK_E, cost_e=COST_E),
                 seconds=round(time.time() - t0))
    json.dump(dict(metrics=m, stats=stats), open(out / 'result.json', 'w'), indent=1)
    json.dump(dict(rankings=rankings, info=info, pool_ids=ids), open(out / 'rankings.json', 'w'))
    print(json.dumps(m['overall']), json.dumps(stats), flush=True)

if __name__ == '__main__':
    torch.set_num_threads(8); main(sys.argv[1])
