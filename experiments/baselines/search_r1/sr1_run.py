"""Search-R1 inference (official validation loop, scripts/nq_hotpotqa/evaluate.sh + search_r1/llm_agent/generation.py)
with the search tool replaced by frozen Qwen3-Embedding-0.6B dense retrieval over the test corpus.

Faithful to the official val loop: greedy (do_sample=False), max_turns=4 search turns + 1 final rollout without search,
topk=3, max_response_length=500 per turn, max_obs_length=500, max_prompt_length=4096 (left-cut rolling context),
responses decoded with skip_special_tokens=True, cut at first </search> (else first </answer>), re-tokenized.
"""
import argparse, json, os, re, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--encoder', default='/root/autodl-tmp/rev/round19/model')
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--gpu_util', type=float, default=0.62)
args = ap.parse_args()

MAX_TURNS, TOPK, MAX_RESP, MAX_OBS, MAX_PROMPT = 4, 3, 500, 500, 4096
QUERY_PROMPT = 'Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: '
norm = lambda t: ' '.join(str(t or '').split())

os.makedirs(args.out, exist_ok=True)
questions = json.load(open(f'{args.data}/questions.json'))
if args.limit: questions = questions[:args.limit]
corpus = json.load(open(f'{args.data}/corpus.json'))

from vllm import LLM, SamplingParams
llm = LLM(model=args.model, dtype='bfloat16', gpu_memory_utilization=args.gpu_util, max_model_len=MAX_PROMPT + MAX_RESP + 16,
          seed=0, enable_prefix_caching=True)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0, max_tokens=MAX_RESP, stop=['</search>'], include_stop_str_in_output=True,
                    skip_special_tokens=True)

from transformers import AutoTokenizer, AutoModel
etok = AutoTokenizer.from_pretrained(args.encoder, padding_side='left', local_files_only=True)
enc = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.float16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
D = torch.as_tensor(np.load(f'{args.data}/doc_emb.npy'), device='cuda', dtype=torch.float16)
assert D.shape[0] == len(corpus), (D.shape, len(corpus))


@torch.no_grad()
def search(queries):
    out = []
    for s in range(0, len(queries), 32):
        b = etok([QUERY_PROMPT + norm(q) for q in queries[s:s + 32]], padding=True, truncation=True, max_length=32768,
                 return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            h = enc(**b).last_hidden_state[:, -1]
        q = torch.nn.functional.normalize(h, p=2, dim=1).half()
        out += (q @ D.T).topk(TOPK, dim=1).indices.tolist()
    return out


def passages2string(ids):
    # official format; wiki-18 'contents' is '"Title"\ntext', so the title appears quoted
    return ''.join(f'Doc {i + 1}(Title: "{corpus[d]["title"]}") {corpus[d]["text"]}\n' for i, d in enumerate(ids))


def postprocess(resp):
    if '</search>' in resp: return resp.split('</search>')[0] + '</search>'
    if '</answer>' in resp: return resp.split('</answer>')[0] + '</answer>'
    return resp


def parse(resp):
    m = re.search(r'<(search|answer)>(.*?)</\1>', resp, re.DOTALL)
    return (m.group(1), m.group(2).strip()) if m else (None, '')


def make_prompt(q):
    q = q.strip()
    if q[-1] != '?': q += '?'
    p = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {q}\n"""
    return tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False)


INVALID = '\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
st = [dict(qid=x['qid'], ctx=tok(make_prompt(x['question']), add_special_tokens=False)['input_ids'], active=True,
           text='', searches=[], retrieved=[], answer=None, turns=0) for x in questions]
t0 = time.time()
for step in range(MAX_TURNS + 1):
    final = step == MAX_TURNS
    act = [s for s in st if s['active']]
    if not act: break
    outs = llm.generate([{'prompt_token_ids': s['ctx'][-MAX_PROMPT:]} for s in act], sp, use_tqdm=False)
    resps = [postprocess(o.outputs[0].text) for o in outs]
    actions = [parse(r) for r in resps]
    to_search = [i for i, (a, _) in enumerate(actions) if a == 'search' and not final]
    hits = dict(zip(to_search, search([actions[i][1] for i in to_search]))) if to_search else {}
    for i, (s, r, (a, c)) in enumerate(zip(act, resps, actions)):
        s['turns'] += 1
        s['text'] += r
        if a == 'answer':
            s['active'] = False; s['answer'] = c; obs = ''
        elif a == 'search':
            if final: obs = ''
            else:
                s['searches'].append(c); s['retrieved'].append(hits[i])
                obs = f'\n\n<information>{passages2string(hits[i]).strip()}</information>\n\n'
        else:
            obs = INVALID
        if final: s['active'] = False; continue
        obs_ids = tok(obs, add_special_tokens=False)['input_ids'][:MAX_OBS] if obs else []
        s['text'] += tok.decode(obs_ids) if obs else ''
        s['ctx'] = s['ctx'] + tok(r, add_special_tokens=False)['input_ids'] + obs_ids
    print(f'step {step} active {len(act)} searched {len(to_search)} t={time.time() - t0:.0f}s', flush=True)

rankings = {}
for s in st:
    seen = []
    for ids in s['retrieved']:
        for d in ids:
            if d not in seen: seen.append(d)
    rankings[s['qid']] = seen
json.dump(rankings, open(f'{args.out}/rankings.json', 'w'))
with open(f'{args.out}/trajectories.jsonl', 'w') as f:
    for s in st:
        f.write(json.dumps({k: s[k] for k in ('qid', 'searches', 'retrieved', 'answer', 'turns', 'text')}) + '\n')
print('DONE', args.name, len(rankings), 'mean_distinct', np.mean([len(v) for v in rankings.values()]),
      'mean_searches', np.mean([len(s['searches']) for s in st]), f'{time.time() - t0:.0f}s', flush=True)
