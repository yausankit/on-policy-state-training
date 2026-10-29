"""Stage A (venv_ner, official spacy 3.6.1 + en_core_web_trf 3.6.1): official SpacyNER over passages and questions.
Only change vs official batch_ner: spacy.prefer_gpu() and a fixed pipe batch size (official uses len/max_workers),
streaming instead of materialising all docs."""
import sys, json, time, os
from collections import defaultdict
sys.path.insert(0, os.environ.get('WORK', '.') + '/round69/linearrag/LinearRAG')
sys.path.insert(0, os.path.dirname(__file__))
import spacy
print('gpu', spacy.prefer_gpu(), flush=True)
from src.ner import SpacyNER
from hashlib import md5
compute_mdhash_id = lambda c, prefix="": prefix + md5(c.encode()).hexdigest()  # = src.utils.compute_mdhash_id
from common69 import passages, load_questions
name, out = sys.argv[1], sys.argv[2]
texts, perm = passages(name)
ner = SpacyNER('en_core_web_trf')
t0 = time.time()
hids = [compute_mdhash_id(t, prefix='passage-') for t in texts]
p2e = {}; s2e = defaultdict(list)
for i, doc in enumerate(ner.spacy_model.pipe(texts, batch_size=int(os.environ.get('NER_BS', '64')))):
    a, b = ner.extract_entities_sentences(doc, hids[i])
    p2e.update(a)
    for s, ents in b.items():
        for e in ents:
            if e not in s2e[s]: s2e[s].append(e)
    if i % 20000 == 0: print(i, len(texts), f'{time.time()-t0:.0f}s', flush=True)
tp = time.time() - t0
qs = load_questions(name)
qner = {qid: sorted(ner.question_ner(q)) for qid, q, _, _ in qs}
json.dump({'passage_hash_id_to_entities': p2e, 'sentence_to_entities': s2e, 'question_entities': qner,
           'ner_seconds': tp, 'n_passages': len(texts)}, open(out, 'w'))
print('done', tp, len(p2e), len(s2e), flush=True)
