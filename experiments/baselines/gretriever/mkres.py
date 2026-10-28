import json,sys
d=sys.argv[1]; names=[('musique','MuSiQue (full, 11,656)'),('hotpot_pool','HotpotQA-Long (pool, 27,744)'),('2wiki_pool','2Wiki-Long (pool, 19,257)')]
L=['# G-Retriever (retrieval only) results, R69','','Official PCST retrieval, topk=3, topk_e=5, cost_e=0.5; frozen Qwen3-Embedding-0.6B. Percent.','',
   '| Test set | Corpus | R@5 | R@10 | C@5 | C@10 |','|---|---|---|---|---|---|']
R={k:json.load(open(f'{d}/{k}/result.json')) for k,_ in names}
for k,n in names:
    o=R[k]['metrics']['overall']; L.append(f"| {n.split(' (')[0]} | {'full' if k=='musique' else 'pool'} | {o['R@5']:.1f} | {o['R@10']:.1f} | {o['C@5']:.1f} | {o['C@10']:.1f} |")
L+=['','## Graph and fallback statistics','','| Test set | nodes | edges | predicates | self-loops dropped | mean sel. nodes | mean sel. edges | median graph-linked passages | fallback ranks in top-5 | fallback ranks in top-10 (of total) | questions with fallback in top-10 |','|---|---|---|---|---|---|---|---|---|---|---|']
for k,n in names:
    s=R[k]['stats']; L.append(f"| {n} | {s['nodes']} | {s['edges']} | {s['predicates']} | {s['dropped_self_loops']} | {s['mean_sel_nodes']:.1f} | {s['mean_sel_edges']:.1f} | {s['median_graph_passages']:.0f} | {s['fallback_ranks_top5']} | {s['fallback_ranks_top10']} / {s['total_ranks_top10']} | {s['questions_with_any_fallback_top10']} |")
for k,n in names:
    L+=['',f'## By type: {n}','','| Type | n | R@5 | R@10 | C@5 | C@10 |','|---|---|---|---|---|---|']
    for t,m in R[k]['metrics']['by_type'].items(): L.append(f"| {t} | {m['n']} | {m['R@5']:.1f} | {m['R@10']:.1f} | {m['C@5']:.1f} | {m['C@10']:.1f} |")
print('\n'.join(L))
