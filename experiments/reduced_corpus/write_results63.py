"""R63: assemble results.md from pool evals (ours/), HippoRAG 2 metrics and h3 stats."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent
H3 = Path('/Users/qchj/code/HippoRAG/reproduce/hipporag2_repro_20261013/h3')
K = ['R@5', 'R@10', 'C@5', 'C@10']
hip = json.loads((R / 'hipporag_metrics.json').read_text())
rd = lambda p: json.loads(Path(p).read_text())
f1 = lambda x: f'{x:.1f}'
POOLS = [('hotpot_pool', 'HotpotQA-Long pool (27,744 passages, 300 questions)'), ('2wiki_pool', '2Wiki-Long pool (19,257 passages, 300 questions)')]
SEEDS = [42, 43, 44, 45, 46]
lines = ['# R63: HippoRAG 2 and our retrievers on the candidate-pool corpora', '',
         'Pool = all gold passages + top-100 of every question under the frozen encoder (single query). '
         'All retrievers ranked over the same pool corpus; metrics as round54/core54.py::metrics.', '']
for sp, title in POOLS:
    o = lambda n: rd(R / 'ours' / sp / f'{n}.json')
    op = [o(f'MQ-OP_{s}') for s in SEEDS]
    mean = lambda sel: {k: sum(sel(m)[k] for m in op) / len(op) for k in K}
    rows = [('Frozen single query', o('Frozen-SQ_13')), ('Frozen chain', o('Frozen-chain_13')), ('HippoRAG 2', hip[sp]),
            ('Teacher forcing gold prefixes (MQ-MDR:13)', o('MQ-MDR_13')), ('Teacher forcing gold subsets (MQ-RN:14)', o('MQ-RN_14'))]
    lines += [f'## {title}', '', '| Method | R@5 | R@10 | C@5 | C@10 |', '|---|---|---|---|---|']
    for name, m in rows:
        lines.append(f'| {name} | ' + ' | '.join(f1(m['overall'][k]) for k in K) + ' |')
    lines.append('| **On-policy state training (MQ-OP mean of 42-46)** | ' + ' | '.join(f1(v) for v in mean(lambda m: m['overall']).values()) + ' |')
    for s, m in zip(SEEDS, op):
        lines.append(f'| MQ-OP:{s} | ' + ' | '.join(f1(m['overall'][k]) for k in K) + ' |')
    types = sorted(hip[sp]['by_type'])
    lines += ['', f'By type ({", ".join(types)}; R@5 / R@10 / C@5 / C@10):', '', '| Method | ' + ' | '.join(types) + ' |', '|---|' + '---|' * len(types)]
    for name, m in rows:
        lines.append(f'| {name} | ' + ' | '.join(' / '.join(f1(m['by_type'][t][k]) for k in K) for t in types) + ' |')
    lines.append('| On-policy state training (mean 42-46) | ' + ' | '.join(' / '.join(f1(v) for v in mean(lambda m, t=t: m['by_type'][t]).values()) for t in types) + ' |')
    lines.append('')
# OpenIE / LLM / timing
lines += ['## HippoRAG 2 run details', '',
          'LLM: deepseek-flash via relay (thinking disabled), Mac only. File names keep the server config label `qwen-flash` (openie_results_ner_qwen-flash.json); the actual LLM is deepseek-flash.',
          'Embedding: R8 frozen Qwen3-Embedding-0.6B (md5-identical weights). Official HippoRAG 2 code, default BaseConfig, no tuning.',
          'GPU stages (index, candidates) ran on the 5090 (/root/autodl-tmp/h2_pool63) after the GPU policy update ruled out 244 card 0 and all other 244/239 cards were occupied by other users; retrieve ran on CPU on the 5090.', '',
          '| Pool | Passages | Mean entities | Mean triples | OpenIE failures | Filter questions with empty selection (dense fallback) |', '|---|---|---|---|---|---|']
for sp, _ in POOLS:
    st = rd(H3 / f'openie_{sp}_stats.json'); fl = rd(H3 / f'filtered_{sp}.json')
    lines.append(f"| {sp} | {st['passages']} | {st['mean_entities']:.2f} | {st['mean_triples']:.2f} | {st['failures']} | {sum(not x['facts'] for x in fl)} |")
lines += ['', 'OpenIE: 46,949 unique passages across both pools; first pass left 5 failed passages (4 relay 429, 1 parse error), removed and rerun once, 0 failures remain.', '',
          '| LLM stage | Logical calls | Prompt tokens | Completion tokens | Cached prompt tokens |', '|---|---|---|---|---|']
tot = [0, 0, 0, 0]
for name, p in [('OpenIE (both pools, cumulative incl. reruns)', H3 / 'llm_costs.json'), ('Filter hotpot_pool', H3 / 'filters/hotpot_pool/llm_costs.json'), ('Filter 2wiki_pool', H3 / 'filters/2wiki_pool/llm_costs.json')]:
    c = rd(p); v = [c['logical_calls'], c['prompt_tokens'], c['completion_tokens'], c['cached_tokens']]
    tot = [a + b for a, b in zip(tot, v)]
    lines.append(f'| {name} | ' + ' | '.join(f'{x:,}' for x in v) + ' |')
lines.append('| **Total** | ' + ' | '.join(f'{x:,}' for x in tot) + ' |')
lines += ['', '| Pool | OpenIE wall (both pools, Mac) | Index (5090 GPU) | Candidates (5090 GPU) | Filter (Mac) | Retrieve (5090 CPU) | Mean per-question retrieve |', '|---|---|---|---|---|---|---|']
T = rd(R / 'timings.json')
for sp, _ in POOLS:
    t = T[sp]
    lines.append(f"| {sp} | {T['openie_s']/3600:.2f} h | {t['index']:.0f} s | {t['candidates']:.0f} s | {t['filter']:.0f} s | {t['retrieve']:.0f} s | {t['per_q']:.2f} s |")
(R / 'results.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
