"""Build HotpotQA-Long chain candidates by following hyperlinks in HotpotQA's 2017 Wikipedia abstracts.

Runs on 239 (r53env). Each chain starts from an original HotpotQA bridge question's first two pages
(A -> B), then follows hyperlinks B -> C -> ... so that a chain has k = 3, 4 or 5 pages. Every hop keeps
the sentence that carries the link. The question and answer are written later by the LLM (a fact stated
in the last page). Components are page-to-page link edges; splits share no component.
"""
import bz2, collections, glob, json, random, re, urllib.parse
from multiprocessing import Pool
from pathlib import Path

ROOT = Path('/home/dmir/qiuchenjie2023/revision_20260928/round53')
DUMP = ROOT / 'data/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts'
RAW = Path('/home/dmir/qiuchenjie2023/revision_20260928/round17/data_prep/hotpotqa/raw')
OUT = ROOT / 'claude/hotpotqa'
LINK = re.compile(r'<a href="([^"]+)">')
TARGET = {'test': 1500, 'dev': 1250, 'train': 5200}
MAX_SENT = 6


def load_file(path):
    pages = {}
    with bz2.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            sents = json.loads(r['text']) if isinstance(r['text'], str) else r['text']
            linked = json.loads(r['text_with_links']) if isinstance(r['text_with_links'], str) else r['text_with_links']
            sents = [s.strip() for s in sents[:MAX_SENT]]
            if len(sents) < 2 or len(' '.join(sents)) < 120:
                continue
            links = []
            for i, s in enumerate(linked[:MAX_SENT]):
                for t in LINK.findall(s):
                    links.append((i, urllib.parse.unquote(t)))
            pages[r['title']] = (sents, links)
    return pages


def load_pages():
    files = sorted(glob.glob(str(DUMP / '*/*.bz2')))
    pages = {}
    with Pool(32) as pool:
        for part in pool.imap_unordered(load_file, files, chunksize=8):
            pages.update(part)
    return pages


def norm(t):
    return re.sub(r'\s*\([^)]*\)$', '', t).lower()


def bridge_starts(records, pages):
    out = []
    for r in records:
        if r['type'] != 'bridge':
            continue
        ts = list(dict.fromkeys(t for t, _ in r['supporting_facts']))
        if len(ts) != 2 or not all(t in pages for t in ts):
            continue
        a, b = ts
        ta, tb = ' '.join(pages[a][0]).lower(), ' '.join(pages[b][0]).lower()
        ab, ba = norm(b) in ta, norm(a) in tb
        if ab == ba:
            continue
        out.append((a, b) if ab else (b, a))
    return out


STOP = {'of', 'the', 'and', 'in', 'on', 'at', 'for', 'de', 'la', 'le', 'von', 'van', 'a', 'an', 'to', 'du', 'des', 'del', 'di'}


def proper(title):
    """Named-entity pages only: every content word capitalised (or a number); concept pages are skipped."""
    words = re.sub(r'\s*\([^)]*\)$', '', title).split()
    return bool(words) and all(w[0].isupper() or w[0].isdigit() or w.lower() in STOP for w in words)


def link_sentence(pages, src, dst):
    for i, t in pages[src][1]:
        if t == dst:
            return pages[src][0][i]
    return None


def build(starts, pages, indeg, k, n, rng, banned, used_edges):
    res, used_start = [], collections.Counter()
    rng.shuffle(starts)
    for a, b in starts * 3:
        if len(res) >= n:
            break
        if used_start[(a, b)] >= 2 or (a, b) in banned:
            continue
        path = [a, b]
        sentence_ab = link_sentence(pages, a, b)
        if sentence_ab is None:
            # the original question links A to B by mention, not necessarily by hyperlink: keep the
            # sentence of A that mentions B
            nb = norm(b)
            sentence_ab = next((s for s in pages[a][0] if nb in s.lower()), None)
            if sentence_ab is None:
                continue
        link_sents = [sentence_ab]
        ok = True
        while len(path) < k:
            cur = path[-1]
            per_sentence = collections.Counter(i for i, _ in pages[cur][1])
            opts = [(i, t) for i, t in pages[cur][1] if per_sentence[i] < 5   # skip list-like sentences
                    and t in pages and t not in path and indeg[t] <= 300 and (cur, t) not in banned and proper(t)
                    and not re.fullmatch(r'\d{3,4}|.*\(disambiguation\)|List of .*', t)]
            if not opts:
                ok = False
                break
            i, t = rng.choice(opts)
            link_sents.append(pages[cur][0][i])
            path.append(t)
        if not ok:
            continue
        used_start[(a, b)] += 1
        hops = []
        for j, p in enumerate(path):
            hops.append(dict(gold_title=p, gold_text=' '.join(pages[p][0]),
                             link_sentence=link_sents[j] if j < len(link_sents) else None,
                             link_to=path[j + 1] if j + 1 < len(path) else None))
        res.append(dict(kind='hchain', hops_n=k, start=re.sub(r'\s*\([^)]*\)$', '', a),
                        hidden=[re.sub(r'\s*\([^)]*\)$', '', p) for p in path[1:]], hops=hops))
        used_edges.update(zip(path, path[1:]))
    return res


def components(c):
    t = [h['gold_title'] for h in c['hops']]
    return set(zip(t, t[1:]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = load_pages()
    indeg = collections.Counter(t for _, links in pages.values() for _, t in links)
    print('pages', len(pages), flush=True)
    dev_starts = bridge_starts(json.load(open(RAW / 'dev.json')), pages)
    train_starts = bridge_starts(json.load(open(RAW / 'train.json')), pages)
    print('bridge starts dev', len(dev_starts), 'train', len(train_starts), flush=True)
    splits = {}
    banned = set()
    for split, starts, seed in (('test', dev_starts, 0), ('dev', train_starts, 100), ('train', train_starts, 200)):
        used = set()
        cands = []
        for k in (3, 4, 5):
            cands += build(list(starts), pages, indeg, k, TARGET[split] // 3, random.Random(seed + k), banned, used)
        splits[split] = cands
        banned |= used
        print(split, len(cands), collections.Counter(c['hops_n'] for c in cands), flush=True)
    comp = {s: set().union(*(components(c) for c in cs)) for s, cs in splits.items()}
    print('component overlap train-dev', len(comp['train'] & comp['dev']), 'train-test',
          len(comp['train'] & comp['test']), 'dev-test', len(comp['dev'] & comp['test']), flush=True)
    for split, cands in splits.items():
        with open(OUT / f'{split}_candidates.jsonl', 'w') as f:
            for i, c in enumerate(cands):
                c['cid'] = f'hpl-{split}-{i:05d}'
                f.write(json.dumps(c, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
