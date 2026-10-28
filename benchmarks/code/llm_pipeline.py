"""Write questions for candidate chains and run the anti-shortcut filters.

Reads DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL from the environment.

Usage: python3 llm_pipeline.py <candidates.jsonl> <results.jsonl> [--limit N]
Resumable: candidates whose cid is already in results.jsonl are skipped.
"""
import argparse, json, os, random, re, string, threading, time, unicodedata, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = os.environ['DEEPSEEK_BASE_URL'] + '/chat/completions'
GEN, CHECK = 'deepseek-flash', 'deepseek-flash'
USAGE = {'calls': 0, GEN: [0, 0]}
LOCK = threading.Lock()


def chat(model, prompt, max_tokens, think=True):
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}]}
    if model.startswith('gpt-'):
        payload['max_completion_tokens'] = max_tokens        # GPT models take max_completion_tokens, default temperature
    else:
        payload.update(max_tokens=max_tokens, temperature=0)
    if not think and not model.startswith('gpt-'):
        payload['thinking'] = {'type': 'disabled'}      # checks need no long reasoning
    body = json.dumps(payload).encode()
    for attempt in range(8):
        req = urllib.request.Request(BASE, data=body, headers={
            'Authorization': 'Bearer ' + os.environ['DEEPSEEK_API_KEY'], 'Content-Type': 'application/json'})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            with LOCK:
                USAGE['calls'] += 1
                u = USAGE.setdefault(model, [0, 0])
                u[0] += r['usage']['prompt_tokens']
                u[1] += r['usage']['completion_tokens']
            return (r['choices'][0]['message'].get('content') or '').strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            time.sleep(min(60, 2 ** attempt + random.random()))
    return ''


def norm(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not p or not g or not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


SAME = """Question: {question}
Reference answer: {gold}
Candidate answer: {pred}
Does the candidate refer to the same entity, date or value as the reference (allowing aliases, translations, abbreviations or different date formats)? Reply "yes" or "no" only."""


def same(question, pred, gold):
    a, b = norm(pred), norm(gold)
    if not a or a == 'unknown':
        return False
    if a == b or f1(pred, gold) >= 0.8 or (len(b) > 3 and b in a):
        return True
    return norm(chat(GEN, SAME.format(question=question, gold=gold, pred=pred), 64, think=False)).startswith('yes')


def correct(question, pred, gold):
    return same(question, pred, gold)


def wrong(question, pred, gold):
    return not same(question, pred, gold)


WRITE_CHAIN = """You are writing one question for a multi-hop question answering dataset.
Reasoning path (each step: subject --relation--> object):
{path}
Write ONE fluent, natural English question that starts from "{start}" and, by following every relation in order, asks for the final object.
Rules:
- The only entity name allowed in the question is "{start}".
- Do NOT mention any of these names or any part of them: {forbidden}.
- Use every relation in the path, in order (e.g. "the mother of the director of film X").
- Phrase it naturally: prefer kinship words (e.g. "paternal grandfather" for father of father, "mother-in-law") and ordinary wording over repeating "the X of the X of".
- The question must have exactly one answer: "{answer}".
Output only the question."""

WRITE_COMP = """You are writing one question for a multi-hop question answering dataset.
Entities: {entities}
For each entity, take its {relation}; then compare the {attribute} of those {relation}s.
Ask which of the listed entities has the {relation} with the {direction} {attribute}.
Rules:
- Mention all listed entities by name, and only them.
- Do NOT mention any of these names: {forbidden}.
- The answer must be one of the listed entities: "{answer}".
Output only the question."""

WRITE_HCHAIN = """You are writing one question for a multi-hop question answering dataset.
Below is a chain of Wikipedia pages. Each page links to the next one through the quoted sentence.
{chain}
Final page "{last}":
{last_text}

Write ONE fluent, natural English question that starts from "{start}", follows every link in order, and asks for one specific fact stated in the final page (a name, date, place or number).
Rules:
- The only entity name allowed in the question is "{start}".
- Do NOT mention any of these names or any part of them: {forbidden}.
- Describe each intermediate entity briefly, only through its relation to the previous one (e.g. "the label that released it", "the city where that label is based").
- Write it as a question a curious person would naturally ask, at most 30 words, with "{start}" as the grammatical subject or object (do NOT open with "For {start}," or "In {start},").
- Never mention links, pages, Wikipedia, lists, "according to", "mentions", or "listed"; avoid stacking more than two relative clauses.
- The answer must be a specific name, date, place or number copied from the final page (not a generic noun), and the question must have exactly one answer.
Output JSON only: {{"question": "...", "answer": "..."}}"""

ANSWER = """{context}Question: {question}
Answer with a short phrase only (an entity name, a date, or a place). If you cannot determine the answer, reply "unknown"."""

UNIQUE = """Passages:
{passages}
Question: {question}
Proposed answer: {answer}
Is the question unambiguous, answerable from the passages, and is the proposed answer its only correct answer? Reply "yes" or "no" only."""


NATURAL = """Rate this question for a question-answering dataset on a 1-5 scale.
5 = reads like a question a curious person would naturally ask; clear and easy to parse.
4 = natural with minor awkwardness.
3 = understandable but clunky, convoluted, or oddly framed.
2 = hard to parse, meta wording (mentions pages, links, lists), or stacked clauses.
1 = broken or nonsensical.
Question: {question}
Reply with the digit only."""


REWRITE = """Rewrite this question so it reads like a question a curious person would naturally ask, without changing what it asks or its answer.
Keep it at most 30 words, keep "{start}" as the only named entity, and do not add any other names.
Do NOT mention any of these names: {forbidden}.
Question: {question}
Output only the rewritten question."""


def passages_text(hops):
    seen, out = set(), []
    for h in hops:
        if h['gold_title'] not in seen:
            seen.add(h['gold_title'])
            out.append(f"Title: {h['gold_title']}\n{h['gold_text']}")
    return out


def context_block(paras):
    return ''.join(f'Passage {i + 1}:\n{p}\n\n' for i, p in enumerate(paras))


def leaks(question, names):
    q = norm(question)
    return [n for n in names if norm(n) and re.search(r'\b' + re.escape(norm(n)) + r'\b', q)]


def process(c):
    rng = random.Random(c['cid'])
    res = dict(cid=c['cid'], kind=c['kind'], hops_n=c['hops_n'], answer=c.get('answer'))
    if c['kind'] == 'hchain':
        forbidden = list(c['hidden'])
        chain = '\n'.join(f'{i + 1}. "{h["gold_title"]}" -> "{h["link_to"]}": "{h["link_sentence"]}"'
                          for i, h in enumerate(c['hops'][:-1]))
        last = c['hops'][-1]
        prompt = WRITE_HCHAIN.format(chain=chain, last=last['gold_title'], last_text=last['gold_text'],
                                     start=c['start'], forbidden=', '.join(f'"{x}"' for x in forbidden))
        obj = {}
        for _ in range(3):
            raw = chat(GEN, prompt, 4096)
            m = re.search(r'\{.*\}', raw, re.S)
            try:
                obj = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                obj = {}
            if obj.get('question') and obj.get('answer'):
                break
        q, c['answer'] = str(obj.get('question', '')).strip(), str(obj.get('answer', '')).strip()
        res['answer'] = c['answer']
        forbidden = forbidden + [c['answer']] if c['answer'] else forbidden
    elif c['kind'] == 'chain':
        path = '\n'.join(f"{i + 1}. {h['subject']} --{h['relation']}--> {h['object']}" for i, h in enumerate(c['hops']))
        forbidden = c['hidden'] + [c['answer']]
        prompt = WRITE_CHAIN.format(path=path, start=c['start'], forbidden=', '.join(f'"{x}"' for x in forbidden),
                                    answer=c['answer'])
    else:
        forbidden = c['hidden']
        prompt = WRITE_COMP.format(entities=', '.join(c['entities']), relation=c['relation'],
                                   attribute=c['attribute'], direction=c['direction'],
                                   forbidden=', '.join(f'"{x}"' for x in forbidden), answer=c['answer'])
    if c['kind'] != 'hchain':
        q = ''
        for _ in range(3):
            q = chat(GEN, prompt, 4096).strip().strip('"')
            if q:
                break
    res['question'] = q
    checks = res['checks'] = {}
    checks['leak'] = not leaks(q, forbidden) and bool(q)
    if not checks['leak']:
        return res
    gold = passages_text(c['hops'])
    order = gold[:]
    rng.shuffle(order)
    pred = chat(CHECK, ANSWER.format(context=context_block(order), question=q), 4096)
    checks['gold_answerable'] = correct(q, pred, c['answer'])
    res['pred_gold'] = pred
    if not checks['gold_answerable']:
        return res
    pred = chat(CHECK, ANSWER.format(context='', question=q), 128, think=False)
    res['tags'] = {'closed_book_answerable': not wrong(q, pred, c['answer'])}   # tag only, not a filter
    res['pred_closed'] = pred
    if c['kind'] in ('chain', 'hchain'):
        partial = [passages_text(c['hops'][-1:])]
        drop = rng.randrange(len(c['hops']) - 1)
        partial.append(passages_text([h for i, h in enumerate(c['hops']) if i != drop]))
    else:
        partial = []
    res['pred_partial'] = []
    ok = True
    for paras in partial:
        pred = chat(CHECK, ANSWER.format(context=context_block(paras), question=q), 128, think=False)
        res['pred_partial'].append(pred)
        ok = ok and wrong(q, pred, c['answer'])
    res['tags']['llm_partial_answerable'] = not ok   # tag only; hop skipping is checked by retrieval

    verdict = chat(CHECK, UNIQUE.format(passages='\n\n'.join(order), question=q, answer=c['answer']), 2048)
    checks['unique_answer'] = norm(verdict).startswith('yes')
    if not checks['unique_answer']:
        return res
    score = re.search(r'[1-5]', chat(GEN, NATURAL.format(question=q), 8, think=False) or '')
    res['naturalness'] = int(score.group(0)) if score else 0
    checks['natural'] = res['naturalness'] >= 3          # 3 = understandable; sampling prefers 4-5
    if res['naturalness'] == 3:
        start = c.get('start') or ', '.join(c.get('entities', []))
        q2 = chat(GEN, REWRITE.format(start=start, forbidden=', '.join(f'"{x}"' for x in forbidden), question=q), 4096).strip().strip('"')
        ok2 = bool(q2) and not leaks(q2, forbidden)
        if ok2:
            pred2 = chat(CHECK, ANSWER.format(context=context_block(order), question=q2), 4096)
            ok2 = correct(q2, pred2, c['answer'])
        if ok2:
            v2 = chat(CHECK, UNIQUE.format(passages='\n\n'.join(order), question=q2, answer=c['answer']), 2048)
            ok2 = norm(v2).startswith('yes')
        if ok2:
            s2 = re.search(r'[1-5]', chat(GEN, NATURAL.format(question=q2), 8, think=False) or '')
            ok2 = bool(s2) and int(s2.group(0)) >= 4
        if ok2:
            res['question_original'], res['question'], q = q, q2, q2
            res['naturalness'] = int(s2.group(0)); checks['natural'] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidates')
    ap.add_argument('results')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--target-passed', type=int, default=0, help='stop once this many candidates have passed')
    a = ap.parse_args()
    cands = [json.loads(l) for l in open(a.candidates)]
    # interleave question types in a fixed random order, so stopping early keeps the type mix
    by = {}
    for c in cands:
        by.setdefault((c['kind'], c['hops_n']), []).append(c)
    rng0 = random.Random(0)
    for v in by.values():
        rng0.shuffle(v)
    cands = [v[i] for i in range(max(map(len, by.values()))) for v in by.values() if i < len(v)]
    if a.limit:
        cands = cands[:a.limit]
    prev = [json.loads(l) for l in open(a.results)] if Path(a.results).exists() else []
    done = {r['cid'] for r in prev}
    n_passed = sum(r['passed'] for r in prev)
    todo = [c for c in cands if c['cid'] not in done]
    print(f'{len(todo)} to process ({len(done)} already done)', flush=True)
    t0 = time.time()
    stop = threading.Event()

    def guarded(c):
        return None if stop.is_set() else process(c)

    with open(a.results, 'a') as out, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(guarded, todo)):
            if r is None:
                continue
            r['passed'] = all(r.get('checks', {}).get(k) for k in
                              ('leak', 'gold_answerable', 'unique_answer', 'natural'))
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
            out.flush()
            n_passed += r['passed']
            if a.target_passed and n_passed >= a.target_passed:
                stop.set()
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(todo)} {time.time() - t0:.0f}s usage={USAGE}', flush=True)
    print('final usage', USAGE, flush=True)
    usage_file = Path(a.results).with_suffix('.usage.json')
    prev = json.loads(usage_file.read_text()) if usage_file.exists() else {'calls': 0, GEN: [0, 0]}
    for k in {GEN, CHECK}:
        prev[k] = [prev[k][0] + USAGE[k][0], prev[k][1] + USAGE[k][1]]
    prev['calls'] += USAGE['calls']
    usage_file.write_text(json.dumps(prev))


if __name__ == '__main__':
    main()
