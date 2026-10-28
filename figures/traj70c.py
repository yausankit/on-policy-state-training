"""Anchor-projected chain trajectories (3- and 4-hop MuSiQue dev). Each query vector f(s_t) is placed at the
softmax(cos/tau)-weighted mean of anchor positions: gold passages g1..gn on the top row, retrieved non-gold passages on
the bottom row; tau = 0.05 (training temperature). Usage: traj70c.py <out_dir> sheet|final [ranks...]"""
import json, sys, os, textwrap
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
out = Path(sys.argv[1]); mode = sys.argv[2]; BASE = os.environ['R70_BASE']; TAU = 0.05
Hd = np.load(BASE + '/round29/inputs/design/doc_emb.npy', mmap_mode='r'); O = np.load(out / 'own70c.npz')
titles = [r['title'] for r in json.load(open(BASE + '/round29/inputs/design/corpus.json'))]
qtext = {r['id']: r['question'] for r in json.load(open(BASE + '/round29/inputs/design/raw.json'))}
TF, OP, GOLD, INK = '#d0645c', '#3a66c9', '#3f8f5e', '#24304a'
rows = []
for i, q in enumerate(O['qids']):
    g = [int(d) for d in O['chain'][i] if d >= 0]; w = {n: [int(d) for d in O[f'{n}_w'][i]] for n in ('TF', 'OP')}
    found = {n: sum(d in g for d in w[n]) for n in w}
    if not (found['OP'] == len(g) and found['TF'] < len(g)): continue
    wrong = list(dict.fromkeys(d for n in ('TF', 'OP') for d in w[n] if d not in g))
    anchors = g + wrong; A = np.asarray(Hd[anchors], np.float32)
    pos = np.array([[0.1 + 0.8 * k / max(len(g) - 1, 1), 1.0] for k in range(len(g))] + [[0.1 + 0.8 * (k + .5) / max(len(wrong), 1), 0.0] for k in range(len(wrong))])
    P, tw, ww = {}, {}, {}
    for n in ('TF', 'OP'):
        Q = O[f'{n}_q'][i]; S = Q @ A.T / TAU; W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True); P[n] = W @ pos
        hist = []; tw[n] = []; ww[n] = []
        for t in range(4):
            miss = [d for d in g if d not in hist]
            if miss: tw[n].append(W[t, anchors.index(miss[0])])
            ww[n].append(W[t, len(g):].sum()); hist.append(w[n][t])
    score = float(np.mean(tw['OP']) - np.mean(tw['TF']) + np.mean(ww['TF']) - np.mean(ww['OP']))
    rows.append(dict(i=i, q=str(q), g=g, w=w, wrong=wrong, pos=pos, P=P, found=found, score=score, hops=len(g)))
rows.sort(key=lambda r: -r['score'])
def panel(ax, r, full=True):
    g, wrong, pos = r['g'], r['wrong'], r['pos']
    ax.add_patch(FancyBboxPatch((0.0, 0.9), 1.0, 0.2, boxstyle='round,pad=0,rounding_size=0.04', fc='#eef6f1', ec='none', zorder=0.5))
    if wrong: ax.add_patch(FancyBboxPatch((0.0, -0.1), 1.0, 0.2, boxstyle='round,pad=0,rounding_size=0.04', fc='#fbefee', ec='none', zorder=0.5))
    for k, d in enumerate(g):
        ax.scatter(*pos[k], marker='s', s=150, color=GOLD, edgecolor='white', linewidth=1, zorder=4)
        ax.text(*pos[k], f'g{k + 1}', ha='center', va='center', color='white', fontsize=6.6, fontweight='bold', zorder=5)
        if full: ax.text(pos[k][0], 1.065 + 0.085 * (k % 2), textwrap.shorten(titles[d], 15, placeholder='…'), ha='center', va='bottom', fontsize=5.8, color=GOLD)
    for k, d in enumerate(wrong):
        by = [n for n in ('TF', 'OP') if d in r['w'][n]]; col = TF if by == ['TF'] else (OP if by == ['OP'] else INK)
        ax.scatter(*pos[len(g) + k], marker='X', s=70, color=col, edgecolor='white', linewidth=0.7, zorder=4)
        if full: ax.text(pos[len(g) + k][0], -0.065 - 0.085 * (k % 2), textwrap.shorten(titles[d], 17, placeholder='…'), ha='center', va='top', fontsize=5.8, color=col)
    for n, col, ls in (('TF', TF, (0, (3, 2))), ('OP', OP, '-')):
        pts = r['P'][n]
        for a, b in zip(pts, pts[1:]):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10, color=col, lw=1.7, linestyle=ls, zorder=3, shrinkA=6, shrinkB=6))
        hist = []
        for t, xy in enumerate(pts, 1):
            w = r['w'][n][t - 1]; hit = w in g and w not in hist; hist.append(w)
            ax.scatter(*xy, s=118, color=GOLD if hit else 'white', edgecolor=col, linewidth=1.8, zorder=6)
            ax.text(*xy, str(t), ha='center', va='center', fontsize=6.8, color='white' if hit else col, fontweight='bold', zorder=7)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.27, 1.25); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
if mode == 'sheet':
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for k, (ax, r) in enumerate(zip(axes.ravel(), rows)):
        panel(ax, r, full=False); ax.set_title(f"#{k} {r['hops']}hop s={r['score']:.2f} TF{r['found']['TF']} OP{r['found']['OP']}\n" + textwrap.shorten(qtext[r['q']], 70), fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'sheet_c.png', dpi=70); print(len(rows))
else:
    pick = [rows[int(k)] for k in sys.argv[3:]]
    for x in range(len(pick)):
        for y in range(x):
            if set(pick[x]['g']) & set(pick[y]['g']): print('shared gold', sys.argv[3 + y], sys.argv[3 + x])
    nr = (len(pick) + 2) // 3; nc = min(3, len(pick))
    fig, axes = plt.subplots(nr, nc, figsize=(8.6, 2.45 * nr))
    for ax, r in zip(np.ravel(axes), pick):
        panel(ax, r); ax.set_title(textwrap.fill(qtext[r['q']], 58), fontsize=6.1, color=INK, loc='left', pad=1)
    h = [plt.Line2D([], [], color=TF, ls=(0, (3, 2)), lw=1.6, marker='o', mfc='white', label='teacher forcing: query at step t'),
         plt.Line2D([], [], color=OP, lw=1.6, marker='o', mfc='white', label='ours: query at step t'),
         plt.Line2D([], [], color=OP, marker='o', ls='', mfc=GOLD, mew=1.6, markersize=7, label='step retrieves a gold passage'),
         plt.Line2D([], [], color=GOLD, marker='s', ls='', label='gold passage (hop order)'),
         plt.Line2D([], [], color=TF, marker='X', ls='', label='non-gold, retrieved by teacher forcing'), plt.Line2D([], [], color=OP, marker='X', ls='', label='… by ours'), plt.Line2D([], [], color=INK, marker='X', ls='', label='… by both')]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=5.9, bbox_to_anchor=(0.5, -0.005), handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.06, wspace=0.035, hspace=0.2); fig.savefig(out / 'fig_traj.pdf', bbox_inches='tight', pad_inches=0.02)
    print(json.dumps([dict(qid=r['q'], hops=r['hops'], found=r['found'], score=round(r['score'], 3)) for r in pick]))
