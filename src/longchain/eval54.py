"""R54 test evaluation: Frozen-SQ (single query), Frozen-chain, and trained RN / MDR / OP-refresh (chain, T=8,
V** readout) on the test split. Writes rankings and metrics (overall and by question type).
Usage: eval54.py --dataset 2wiki_long --condition Frozen-SQ|Frozen-chain|RN|MDR|OP-refresh [--seed 13]"""
import argparse
from core54 import ROOT, Encoder, save, seed_all, load_split, load_docs, chain_retrieve, metrics

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True)
ap.add_argument('--condition', required=True)
ap.add_argument('--seed', type=int, default=13)
ap.add_argument('--T', type=int, default=8)
a = ap.parse_args()
seed_all(a.seed)
out = ROOT / 'evaluations' / a.dataset / a.condition / str(a.seed)
if (out / 'metrics.json').exists():
    raise SystemExit(0)
qs, raw, gold, corp = load_split(a.dataset, 'test')
D = load_docs(a.dataset, 'test')
if a.condition.startswith('Frozen'):
    enc = Encoder(frozen=True)
else:
    enc = Encoder(str(ROOT / 'lora' / a.dataset / a.condition / str(a.seed) / 'adapter'))
enc.model.eval()
rankings, _ = chain_retrieve(enc, qs, corp, D, a.T, single=(a.condition == 'Frozen-SQ' or a.condition.startswith('ABL-SQ')))
m = metrics(rankings, gold, raw)
save(out / 'rankings.json', rankings)
save(out / 'metrics.json', m)
print(a.dataset, a.condition, {k: round(v, 1) for k, v in m['overall'].items()}, flush=True)
for t, v in sorted(m['by_type'].items()):
    print('   ', t, v['n'], {k: round(x, 1) for k, x in v.items() if k != 'n'}, flush=True)
