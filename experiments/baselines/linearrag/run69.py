"""Stage B (venv_ret): official LinearRAG indexing + retrieval with the frozen Qwen3-Embedding-0.6B.
Official classes/methods are used unchanged except the overrides marked DEVIATION (memory/speed only, same math)."""
import sys, os, json, time, math, gc
import numpy as np, torch
sys.path.insert(0, '/home/amax/revision_20260928/round69/linearrag/LinearRAG')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types
_ner = types.ModuleType('src.ner'); _ner.SpacyNER = None; sys.modules['src.ner'] = _ner  # NER precomputed in stage A with the official spacy env
from src.config import LinearRAGConfig
from src import LinearRAG as LR
from src.utils import compute_mdhash_id
from transformers import AutoTokenizer, AutoModel
from common69 import DATA, CFG, passages, load_questions
QP = 'Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: '
MODEL = '/home/amax/revision_20260928/round8/encoder/model'

class QwenST:
    """SentenceTransformer-like encode(): str -> query (instruction prefix), list -> documents/entities/sentences (no prefix).
    In official LinearRAG the only str call is the question embedding in retrieve()."""
    def __init__(self, bs=128):
        self.tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
        self.m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval(); self.bs = bs
    @torch.no_grad()
    def _enc(self, texts):
        texts = [' '.join(str(t or '').split()) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i])); out = np.empty((len(texts), 1024), np.float32)
        i = 0
        while i < len(order):
            L = len(texts[order[i]]); bs = max(4, min(self.bs * 4, int(self.bs * 400 / max(L, 50))))
            idx = order[i:i + bs]
            t = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=8192, return_tensors='pt').to('cuda')
            h = self.m(**t).last_hidden_state[:, -1]
            out[idx] = torch.nn.functional.normalize(h.float(), p=2, dim=1).cpu().numpy(); i += bs
        return out
    def encode(self, x, normalize_embeddings=True, show_progress_bar=False, batch_size=None):
        if isinstance(x, str): return self._enc([QP + ' '.join(x.split())])[0]
        if len(x) == 0: return np.zeros((0, 1024), np.float32)
        return self._enc(list(x))

class Store:
    """DEVIATION (memory): in-memory replacement of official EmbeddingStore (no parquet, embeddings as one array)."""
    def __init__(self, model, namespace, dtype=np.float32):
        self.m, self.ns, self.dtype = model, namespace, dtype
        self.hash_ids, self.texts, self.embeddings = [], [], np.zeros((0, 1024), dtype)
        self.hash_id_to_text, self.hash_id_to_idx, self.text_to_hash_id = {}, {}, {}
    def insert_text(self, text_list, emb=None):
        d = {}
        for t in text_list: d[compute_mdhash_id(t, prefix=self.ns + '-')] = t
        missing = [h for h in d if h not in self.hash_id_to_idx]
        texts = [d[h] for h in missing]
        if emb is not None: e = emb
        else:
            e = np.zeros((len(texts), 1024), self.dtype)
            for a in range(0, len(texts), 200000): e[a:a + 200000] = self.m.encode(texts[a:a + 200000])
        base = len(self.hash_ids)
        self.hash_ids += missing; self.texts += texts
        self.embeddings = np.concatenate([self.embeddings, e.astype(self.dtype)]) if base else e.astype(self.dtype)
        for k, (h, t) in enumerate(zip(missing, texts)):
            self.hash_id_to_idx[h] = base + k; self.hash_id_to_text[h] = t; self.text_to_hash_id[t] = h
    def get_hash_id_to_text(self): return dict(self.hash_id_to_text)

class LinearRAG69(LR.LinearRAG):
    def __init__(self, config, ner):
        self.config = config; self.dataset_name = config.dataset_name; self.device = torch.device('cuda')
        self.ner = ner; self.llm_model = None
        self.passage_embedding_store = Store(config.embedding_model, 'passage', np.float16)
        self.entity_embedding_store = Store(config.embedding_model, 'entity', np.float16)
        self.sentence_embedding_store = Store(config.embedding_model, 'sentence', np.float16)
        self.graph = LR.ig.Graph(directed=False)
        self.ner_results_path = '/dev/null'
    def load_existing_data(self, passage_hash_ids):  # precomputed official NER (stage A)
        return self.ner['passage_hash_id_to_entities'], self.ner['sentence_to_entities'], set()
    def save_ner_results(self, *a): pass
    def add_nodes(self):  # DEVIATION (speed): batch add_vertices instead of one add_vertex per node; same nodes/attrs/order
        ent = self.entity_embedding_store.get_hash_id_to_text(); pas = self.passage_embedding_store.get_hash_id_to_text()
        allh = {**ent, **pas}
        self.graph.add_vertices(len(allh), attributes={'name': list(allh.keys()), 'content': list(allh.values())})
        self.node_name_to_vertex_idx = {v: i for i, v in enumerate(self.graph.vs['name'])}
        self.passage_node_indices = [self.node_name_to_vertex_idx[p] for p in set(pas.keys())]
    # --- retrieval-time speed overrides (identical math) ---
    def dense_passage_retrieval(self, question_embedding):  # DEVIATION (speed): GPU matmul
        s = (self.P_gpu @ torch.from_numpy(np.asarray(question_embedding, np.float32)).cuda()).cpu().numpy()
        idx = np.argsort(s)[::-1]; return idx, s[idx].tolist()
    def get_seed_entities(self, question):  # question NER precomputed with the official question_ner (stage A)
        qe = self.q_ents[question]
        if not qe: return [], [], [], []
        qemb = self.config.embedding_model.encode(qe)
        sims = (self.E_gpu @ torch.from_numpy(qemb).cuda().T).cpu().numpy()
        out = ([], [], [], [])
        for j in range(len(qe)):
            b = int(np.argmax(sims[:, j])); h = self.entity_hash_ids[b]
            out[0].append(b); out[1].append(self.entity_embedding_store.hash_id_to_text[h]); out[2].append(h); out[3].append(sims[b, j])
        return out
    def calculate_passage_scores(self, question, question_embedding, actived_entities):  # DEVIATION (speed): lowercase cache
        passage_weights = np.zeros(self.n_nodes)
        idx, sc = self.dense_passage_retrieval(question_embedding)
        sc = LR.min_max_normalize(sc)
        ents = [(self.entity_embedding_store.hash_id_to_text[h].lower(), s, t) for h, (_, s, t) in actived_entities.items()]
        bonus = np.zeros(len(self.lower))
        for el, s, t in ents:
            denom = t if t >= 1 else 1
            for pi, txt in enumerate(self.lower):
                c = txt.count(el)
                if c > 0: bonus[pi] += s * math.log(1 + c) / denom
        for i, pi in enumerate(idx):
            ps = self.config.passage_ratio * sc[i] + math.log(1 + bonus[pi])
            passage_weights[self.p_node[pi]] = ps * self.config.passage_node_weight
        return passage_weights
    def prep(self, questions):
        ps = self.passage_embedding_store
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.passage_hash_ids = list(ps.hash_id_to_text.keys())
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = self.sentence_embedding_store.embeddings
        self.E_gpu = torch.from_numpy(self.entity_embedding_store.embeddings).cuda().float()
        self.P_gpu = torch.from_numpy(ps.embeddings).cuda().float()
        self.vertex_idx_to_node_name = dict(enumerate(self.graph.vs['name']))
        self.n_nodes = self.graph.vcount()
        self.lower = [t.lower() for t in ps.texts]
        self.p_node = [self.node_name_to_vertex_idx[h] for h in ps.hash_ids]
        self.q_ents = questions

def main():
    name, nerpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = CFG[DATA[name]['cfg']]
    texts, perm = passages(name)
    doc = np.load(os.path.join(DATA[name]['dir'], 'doc_emb.npy'), mmap_mode='r')
    qs = load_questions(name)
    ner = json.load(open(nerpath)); qner = ner.pop('question_entities')
    model = QwenST()
    conf = LinearRAGConfig(dataset_name=name, embedding_model=model, spacy_model='en_core_web_trf', llm_model=None,
                           retrieval_top_k=20, **cfg)
    print(conf, flush=True)
    rag = LinearRAG69(conf, ner)
    t0 = time.time()
    pe = np.asarray(doc[perm], np.float16)
    rag.passage_embedding_store.insert_text(texts, emb=pe); del pe; gc.collect()  # DEVIATION: reuse frozen doc_emb.npy
    # official index() minus the parquet/graphml I/O
    ner_p2e, ner_s2e, _ = rag.load_existing_data(None)
    en, sn, p2e, rag.entity_to_sentence, rag.sentence_to_entity = rag.extract_nodes_and_edges(ner_p2e, ner_s2e)
    print('entities', len(en), 'sentences', len(sn), flush=True)
    rag.node_to_node_stats = LR.defaultdict(dict)
    t1 = time.time(); rag.sentence_embedding_store.insert_text(list(sn)); t_sent = time.time() - t1
    t1 = time.time(); rag.entity_embedding_store.insert_text(list(en)); t_ent = time.time() - t1
    print(f'sent emb {t_sent:.0f}s ent emb {t_ent:.0f}s', flush=True)
    es, ss = rag.entity_embedding_store, rag.sentence_embedding_store
    rag.entity_hash_id_to_sentence_hash_ids = {es.text_to_hash_id[e]: [ss.text_to_hash_id[s] for s in v] for e, v in rag.entity_to_sentence.items()}
    rag.sentence_hash_id_to_entity_hash_ids = {ss.text_to_hash_id[s]: [es.text_to_hash_id[e] for e in v] for s, v in rag.sentence_to_entity.items()}
    del ner, ner_s2e, rag.entity_to_sentence, rag.sentence_to_entity, sn, en; gc.collect()
    rag.add_entity_to_passage_edges(p2e); del p2e
    rag.add_adjacent_passage_edges(); rag.augment_graph(); del rag.node_to_node_stats; gc.collect()
    t_index = time.time() - t0
    print('graph', rag.graph.vcount(), rag.graph.ecount(), f'index {t_index:.0f}s', flush=True)
    qmap = {q: qner[qid] for qid, q, _, _ in qs}
    rag.prep(qmap)
    pid2corpus = {h: perm[i] for i, h in enumerate(rag.passage_embedding_store.hash_ids)}
    rankings, t2, nseed = {}, time.time(), 0
    for n, (qid, q, g, ty) in enumerate(qs):
        qemb = model.encode(q)
        si, st, sh, ssc = rag.get_seed_entities(q)
        if len(st):
            nseed += 1
            hids, _ = rag.graph_search_with_seed_entities(q, qemb, si, st, sh, ssc)
            rankings[qid] = [pid2corpus[h] for h in hids[:conf.retrieval_top_k]]
        else:
            idx, _ = rag.dense_passage_retrieval(qemb)
            rankings[qid] = [perm[i] for i in idx[:conf.retrieval_top_k]]
        if n % 50 == 0: print('q', n, f'{time.time()-t2:.0f}s', flush=True)
    json.dump(dict(rankings=rankings, timing=dict(ner_seconds=None, sentence_embed_s=t_sent, entity_embed_s=t_ent,
              index_total_excl_ner_s=t_index, retrieval_s=time.time() - t2), n_seeded=nseed, n=len(qs),
              graph=dict(nodes=rag.graph.vcount(), edges=rag.graph.ecount(), entities=len(rag.entity_hash_ids),
                         sentences=len(rag.sentence_hash_ids)), config={k: v for k, v in vars(conf).items() if k not in ('embedding_model', 'llm_model')}),
              open(out, 'w'))
    print('done', flush=True)
if __name__ == '__main__': main()
