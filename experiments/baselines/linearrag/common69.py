"""Shared dataset loading for the R69 LinearRAG run (no heavy deps)."""
import json, os, random
import numpy as np
R = os.environ.get('WORK', '.') + ''
DATA = {
    'musique': dict(dir=f'{R}/round29/inputs/design', kind='musique', cfg='musique'),
    'hotpot_full': dict(dir=f'{R}/round57/round54/data/hotpotqa_long_ref/inputs/test', kind='long', cfg='hotpotqa'),
    'twowiki_full': dict(dir=f'{R}/round57/round54/data/2wiki_long_ref/inputs/test', kind='long', cfg='2wikimultihop'),
    'hotpot_pool': dict(dir=f'{R}/round57/round54/data/hotpotqa_long_pool/inputs/test', kind='long', cfg='hotpotqa'),
}
# official per-dataset settings from LinearRAG/scripts/run.sh
CFG = {
    'musique': dict(max_iterations=5, passage_ratio=2.0, iteration_threshold=0.1, top_k_sentence=4),
    '2wikimultihop': dict(max_iterations=3, passage_ratio=0.05, iteration_threshold=0.4, top_k_sentence=1),
    'hotpotqa': dict(max_iterations=3, passage_ratio=0.05, iteration_threshold=0.4, top_k_sentence=1),
}
def norm(t): return ' '.join(str(t or '').split())
def load_corpus(name):
    c = json.load(open(os.path.join(DATA[name]['dir'], 'corpus.json')))
    return [norm(v.get('title')) + '\n' + norm(v.get('text') or v.get('paragraph_text') or '') for v in c]
def permutation(n, seed=0):
    p = list(range(n)); random.Random(seed).shuffle(p); return p  # p[pos] = corpus index
def passages(name):
    """LinearRAG passages in shuffled order: f'{pos}:{text}' as in official run.py; returns (texts, perm)."""
    corp = load_corpus(name); perm = permutation(len(corp))
    return [f'{pos}:{corp[ci]}' for pos, ci in enumerate(perm)], perm
def load_questions(name):
    """returns list of (qid, question, gold corpus indices, type)"""
    d = DATA[name]['dir']
    if DATA[name]['kind'] == 'musique':
        raw = json.load(open(os.path.join(d, 'raw.json')))
        c = json.load(open(os.path.join(d, 'corpus.json')))
        key = {(norm(v['title']), norm(v['text'])): i for i, v in enumerate(c)}
        out = []
        for r in raw:
            g = [key[(norm(p['title']), norm(p['paragraph_text']))] for p in r['paragraphs'] if p['is_supporting']]
            out.append((r['id'], r['question'], g, r['id'].split('__')[0]))
        return out
    qs = json.load(open(os.path.join(d, 'questions.json')))
    gold = json.load(open(os.path.join(d, 'ordered_gold.json')))
    raw = {r['_id']: r for r in json.load(open(os.path.join(d, 'raw.json')))}
    return [(q['qid'], q['question'], gold[q['qid']], raw[q['qid']]['type']) for q in qs]
