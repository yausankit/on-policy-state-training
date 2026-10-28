"""Independent re-verification of R53 questions with a second model family (gpt-5.4-mini via the relay).

For each question: (1) answer it from the shuffled gold passages and compare with the reference (normalised
string match, falling back to the deepseek-flash equivalence judge); (2) ask whether the question is
unambiguous with the reference as its only answer. Both must pass.
Usage (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL set): python3 verify53.py <candidates.jsonl> <results.jsonl> <cids.txt> <out.jsonl>
"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import llm_pipeline as lp

VERIFY = 'gpt-5.4-mini'
cand_path, res_path, cid_path, out_path = sys.argv[1:5]
cands = {c['cid']: c for c in map(json.loads, open(cand_path))}
results = {r['cid']: r for r in map(json.loads, open(res_path))}
cids = [l.strip() for l in open(cid_path) if l.strip()]
done = {json.loads(l)['cid'] for l in open(out_path)} if Path(out_path).exists() else set()


def verify(cid):
    c, r = cands[cid], results[cid]
    q, ans = r['question'], r['answer']
    gold = lp.passages_text(c['hops'])
    random.Random(cid + 'v').shuffle(gold)
    pred = lp.chat(VERIFY, lp.ANSWER.format(context=lp.context_block(gold), question=q), 2048)
    answer_ok = lp.correct(q, pred, ans)
    unique = lp.norm(lp.chat(VERIFY, lp.UNIQUE.format(passages='\n\n'.join(gold), question=q, answer=ans), 1024)).startswith('yes')
    return dict(cid=cid, verify_model=VERIFY, verify_pred=pred, verify_answer_ok=answer_ok, verify_unique=unique,
                verified=answer_ok and unique)


todo = [c for c in cids if c not in done]
print(f'{len(todo)} to verify', flush=True)
with open(out_path, 'a') as out, ThreadPoolExecutor(8) as ex:
    for i, row in enumerate(ex.map(verify, todo)):
        out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
        if (i + 1) % 100 == 0:
            print(i + 1, flush=True)
print('done', flush=True)
