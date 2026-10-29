"""Downstream QA on MuSiQue (Table 2, Table 8): the top 5 passages of each retriever go to the same reader
(temperature 0, thinking disabled, max_tokens 64); EM / F1 against the gold answers. QA_MODEL selects the reader
(deepseek-flash or qwen-flash). Responses are cached in qa_cache.jsonl (key = system|qid); a rerun resumes.
Set RANKINGS_DIR to the directory holding each system's ranking_ids.json (see SYSTEMS).
"""
import json, os, re, string, sys, time, collections, threading
from concurrent.futures import ThreadPoolExecutor
import httpx
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

ROOT = os.environ.get("RANKINGS_DIR", "rankings")
REV = ROOT
OUT = os.environ.get("QA_OUT", ".")
MODEL = os.environ.get("QA_MODEL", "deepseek-flash")
CACHE = f"{OUT}/qa_cache_{MODEL}.jsonl"
TOPK = 5
SYSTEMS = {
    "Dense": f"{REV}/dense.json",
    "Teacher forcing (RN)": f"{REV}/teacher_forcing.json",
    "On-policy state training (OP-refresh)": f"{REV}/on_policy.json",
    "Search-R1": f"{REV}/search_r1.json",
    "HippoRAG 2": f"{REV}/hipporag2.json",
    "Q-RAG": f"{REV}/qrag.json",
    "G-Retriever": f"{REV}/gretriever.json",
    "LinearRAG": f"{REV}/linearrag.json",
}
SYS_PROMPT = ("Answer the question using the given passages. Reply with a short answer only "
              "(an entity, date or number), no explanation.")


def load_env():
    # Credentials come from the environment: DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL.
    return os.environ


# SQuAD / MuSiQue normalization
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def build_user(passages, question):
    blocks = [f"Title: {t}\n{x}" for t, x in passages]
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def main():
    data = json.load(open(os.environ.get("MUSIQUE_DIR", "dataset") + "/musique.json"))
    corpus = json.load(open(os.environ.get("MUSIQUE_DIR", "dataset") + "/musique_corpus.json"))
    ids = [r["id"] for r in data]
    assert len(set(ids)) == 1000
    rankings = {}
    for name, p in SYSTEMS.items():
        d = json.load(open(p))
        missing = [q for q in ids if q not in d or len(d[q]) == 0]
        short = sum(1 for q in ids if q in d and 0 < len(d[q]) < TOPK)
        bad = [q for q in ids if q in d and any(not (0 <= int(i) < len(corpus)) for i in d[q][:TOPK])]
        print(f"[cover] {name}: {len(d)} entries, missing={len(missing)}, fewer_than_{TOPK}={short}, bad_idx={len(bad)}")
        assert not missing and not bad, name
        rankings[name] = d

    jobs = []  # (system, qid, user_msg)
    for r in data:
        for name, d in rankings.items():
            ps = [(corpus[int(i)]["title"], corpus[int(i)]["text"]) for i in d[r["id"]][:TOPK]]
            jobs.append((name, r["id"], build_user(ps, r["question"])))
        gold = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p["is_supporting"]]
        jobs.append(("Gold passages (oracle)", r["id"], build_user(gold, r["question"])))

    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try:
                o = json.loads(line)
                cache[(o["system"], o["qid"])] = o
            except Exception:
                pass
    todo = [j for j in jobs if (j[0], j[1]) not in cache]
    print(f"[jobs] total={len(jobs)} cached={len(jobs)-len(todo)} todo={len(todo)}")

    env = load_env()
    client = OpenAI(api_key=env["DEEPSEEK_API_KEY"], base_url=env["DEEPSEEK_BASE_URL"],
                    http_client=httpx.Client(proxy="http://127.0.0.1:7892", timeout=120),
                    max_retries=0)
    lock = threading.Lock()
    fh = open(CACHE, "a")
    done = [0]

    def call(job):
        name, qid, user = job
        delay = 2.0
        for attempt in range(30):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=64,
                    messages=[{"role": "system", "content": SYS_PROMPT},
                              {"role": "user", "content": user}],
                    extra_body={"thinking": {"type": "disabled"}})
                u = resp.usage
                o = {"system": name, "qid": qid,
                     "pred": (resp.choices[0].message.content or "").strip(),
                     "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                     "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0}
                with lock:
                    fh.write(json.dumps(o, ensure_ascii=False) + "\n"); fh.flush()
                    cache[(name, qid)] = o
                    done[0] += 1
                    if done[0] % 200 == 0:
                        print(f"[progress] {done[0]}/{len(todo)}", flush=True)
                return
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                time.sleep(min(delay, 60)); delay *= 1.6
            except APIStatusError as e:
                if e.status_code >= 500 or e.status_code == 429:
                    time.sleep(min(delay, 60)); delay *= 1.6
                else:
                    print(f"[error] {name} {qid} status={e.status_code}", flush=True)
                    return
        print(f"[giveup] {name} {qid}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(call, todo))
    fh.close()

    results = {}
    order = list(SYSTEMS) + ["Gold passages (oracle)"]
    for name in order:
        ems, f1s, pt, ct, n = [], [], 0, 0, 0
        for r in data:
            o = cache.get((name, r["id"]))
            golds = [r["answer"]] + list(r.get("answer_aliases") or [])
            pred = o["pred"] if o else ""
            if o:
                n += 1; pt += o["prompt_tokens"]; ct += o["completion_tokens"]
            ems.append(max(em(pred, g) for g in golds))
            f1s.append(max(f1(pred, g) for g in golds))
        results[name] = {"n_answered": n, "EM": 100 * sum(ems) / len(ems), "F1": 100 * sum(f1s) / len(f1s),
                         "prompt_tokens": pt, "completion_tokens": ct}
    json.dump({"model": MODEL, "top_k": TOPK, "n_questions": len(data), "results": results},
              open(f"{OUT}/qa_results.json", "w"), indent=2)
    lines = ["# Round 67: MuSiQue downstream QA (reader deepseek-flash, top-5 passages)", "",
             "| System | EM | F1 | Answered | Prompt tokens | Completion tokens |",
             "|---|---|---|---|---|---|"]
    for name in order:
        r = results[name]
        lines.append(f"| {name} | {r['EM']:.1f} | {r['F1']:.1f} | {r['n_answered']}/1000 | "
                     f"{r['prompt_tokens']:,} | {r['completion_tokens']:,} |")
    tp = sum(r["prompt_tokens"] for r in results.values()); tc = sum(r["completion_tokens"] for r in results.values())
    lines += ["", f"Totals: prompt {tp:,} tokens, completion {tc:,} tokens.",
              "Oracle row uses all is_supporting paragraphs in listed order (not truncated to 5).",
              "Search-R1 retrieved fewer than 5 distinct passages for 123 questions (75 with 4, 48 with 3); those get all of them."]
    open(f"{OUT}/results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
