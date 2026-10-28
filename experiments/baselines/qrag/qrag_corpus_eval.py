"""Run a trained Q-RAG agent over a full test corpus with its own inference procedure.

Inference is the official one (eval_retriever.py::evaluate_episode): greedy argmax of
Q(s, a) = <policy.state_embed(s), critic.action_embed(a)> over all not-yet-selected chunks,
one chunk per step, state text = question [SEP] chunk_1 [SEP] ... (official QAEnv / stack_memory),
for envs.max_steps steps (never_terminate: no gold-based stopping).
The only change for our setup: the chunk set is the whole test corpus instead of the
20 per-question MuSiQue paragraphs, so chunk embeddings are computed once per corpus.

Ranking = selected chunks in order, then the remaining chunks ranked by the final-step Q-values.

Usage: python qrag_corpus_eval.py <qrag_repo> <run_dir> <dataset_dir> <dataset_name> <out_dir> [--local_check]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('run_dir'); ap.add_argument('data_dir'); ap.add_argument('name'); ap.add_argument('out_dir')
ap.add_argument('--tail', type=int, default=50, help='number of final-step candidates appended after the selected chunks')
ap.add_argument('--ckpt', default='model_best.pt')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--safe_load', action='store_true')
ap.add_argument('--local_dev', default=None, help='jsonl of MuSiQue samples: run on per-question 20-paragraph contexts (equivalence check)')
args = ap.parse_args()

sys.path.insert(0, args.repo)
os.chdir(args.repo)
from omegaconf import OmegaConf
from rl.agents.pqn import PQN
from envs.utils import stack_memory, stack_text_list

cfg = OmegaConf.load(os.path.join(args.run_dir, 'config.yaml'))
test_cfg = OmegaConf.load(os.path.join(args.repo, 'configs/testing.yaml'))
cfg = OmegaConf.merge(cfg, OmegaConf.create({'seed': 42, 'device': args.device}))  # testing.yaml defaults
OmegaConf.resolve(cfg)
torch.manual_seed(42); np.random.seed(42)
torch.set_default_device(args.device)
torch.set_float32_matmul_precision('high')

agent = PQN(cfg.algo)
torch.set_default_device('cpu')  # read the checkpoint into host memory, then copy into the GPU modules
if args.safe_load:
    # tensors-only unpickling (torch weights_only=True): no code from the file is executed; optimizer state is not needed
    ck = torch.load(os.path.join(args.run_dir, args.ckpt), map_location='cpu', weights_only=True)
    agent.critic.load_state_dict(ck['critic'], strict=True)
    agent.policy.load_state_dict(ck['policy'], strict=True)
    agent.v_net_target.load_state_dict(ck['v_net_target'], strict=True)
    agent.action_embed_target.load_state_dict(ck['action_embed_target'], strict=True)
    del ck
else:
    agent.load(os.path.join(args.run_dir, args.ckpt), strict=True)
torch.set_default_device(args.device)
agent.eval()
max_steps = int(cfg.envs.max_steps)
L_act = int(cfg.max_action_length)
L_mem = int(cfg.algo.pqn.hyperparams.max_action_length_in_memory)
sep = cfg.envs.env.separator
assert cfg.envs.positions_processor == 'none' and not cfg.envs.sort_by_index
state_embed = agent.policy.state_embed
action_embed = agent.critic.action_embed
tok_a = agent.action_tokenizer
print(f'max_steps={max_steps} L_act={L_act} L_mem={L_mem} sep={sep!r} dtype={next(action_embed.parameters()).dtype}', flush=True)


@torch.no_grad()
def embed_chunks(texts, bs=256):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.empty(len(texts), action_embed.model.model_dim, dtype=torch.float32)
    for s in range(0, len(order), bs):
        ids = order[s:s + bs]
        batch = stack_text_list([texts[i] for i in ids], tok_a, L_act)
        e = action_embed(**batch, positions=torch.zeros(len(ids)))['rope']
        out[torch.tensor(ids)] = e.float()
        if (s // bs) % 200 == 0:
            print(f'  embedded {s + len(ids)}/{len(texts)}', flush=True)
    return out


class Mem:  # minimal TextMemory stand-in for stack_memory (tokens come from .text only)
    def __init__(self, text):
        self.text = text; self.item_ids = []; self.available_ids = set(); self.available_mask = np.ones(1, dtype=bool)


@torch.no_grad()
def state_vec(texts):
    st = stack_memory([Mem(t) for t in texts], agent.critic.action_embed.tokenizer, max_length=L_mem)
    return state_embed(input_ids=st.input_ids, attention_mask=st.attention_mask).float()


@torch.no_grad()
def run(questions, chunk_texts, chunk_emb, tail):
    """questions: list of str; returns list of (selected, appended, qmax list)."""
    res = []
    B = 32
    for b in range(0, len(questions), B):
        qs = questions[b:b + B]
        states = list(qs)
        sel = [[] for _ in qs]
        qmax = [[] for _ in qs]
        for step in range(max_steps):
            s = state_vec(states)  # [b, D]
            logits = s @ chunk_emb.T  # [b, N]
            for i in range(len(qs)):
                if sel[i]:
                    logits[i, torch.tensor(sel[i])] = logits[i].min() - 1
            a = logits.argmax(-1).tolist()
            for i in range(len(qs)):
                qmax[i].append(float(logits[i, a[i]]))
                sel[i].append(a[i])
                states[i] = sep.join([qs[i]] + [chunk_texts[j] for j in sel[i]])
            if step == max_steps - 1:
                k = min(tail + max_steps, logits.shape[1])
                top = torch.topk(logits, k, dim=1).indices.tolist()
                for i in range(len(qs)):
                    app = [j for j in top[i] if j not in set(sel[i])][:tail]
                    res.append((sel[i], app, qmax[i]))
    return res


os.makedirs(args.out_dir, exist_ok=True)
t0 = time.time()
if args.local_dev:
    # official per-question setting: chunks = title + '. ' + paragraph_text of the 20 paragraphs
    samples = [json.loads(l) for l in open(args.local_dev)]
    out = {}
    for smp in samples:
        texts = [p['title'] + '. ' + p['paragraph_text'] for p in smp['paragraphs']]
        emb = embed_chunks(texts)
        (sel, app, qm), = run([smp['question']], texts, emb.to(args.device), 0)
        out[smp['id']] = sel
    json.dump(out, open(os.path.join(args.out_dir, 'local_dev_pred.json'), 'w'))
    print('local dev done', time.time() - t0)
    sys.exit()

corpus = json.load(open(os.path.join(args.data_dir, 'corpus.json')))
questions = json.load(open(os.path.join(args.data_dir, 'questions.json')))
texts = [c['title'] + '. ' + c['text'] for c in corpus]  # MuSiQue adapter chunk format (training format)
emb_path = os.path.join(args.out_dir, 'chunk_emb.npy')
if os.path.exists(emb_path):
    emb = torch.from_numpy(np.load(emb_path).astype(np.float32))
else:
    emb = embed_chunks(texts)
    np.save(emb_path, emb.numpy().astype(np.float16))
t_emb = time.time() - t0
print(f'corpus {len(texts)} embedded in {t_emb:.0f}s', flush=True)
emb = emb.to(args.device)
t1 = time.time()
res = run([q['question'] for q in questions], texts, emb, args.tail)
t_run = time.time() - t1
selected = {q['qid']: r[0] for q, r in zip(questions, res)}
ranking = {q['qid']: r[0] + r[1] for q, r in zip(questions, res)}
qvals = {q['qid']: r[2] for q, r in zip(questions, res)}
json.dump(selected, open(os.path.join(args.out_dir, 'selected_ids.json'), 'w'))
json.dump(ranking, open(os.path.join(args.out_dir, 'ranking_ids.json'), 'w'))
json.dump(qvals, open(os.path.join(args.out_dir, 'step_qmax.json'), 'w'))
json.dump(dict(name=args.name, n_q=len(questions), n_corpus=len(texts), max_steps=max_steps, tail=args.tail,
               embed_sec=t_emb, retrieve_sec=t_run, ckpt=args.ckpt), open(os.path.join(args.out_dir, 'run_info.json'), 'w'), indent=1)
print('done', args.name, t_emb, t_run, flush=True)
