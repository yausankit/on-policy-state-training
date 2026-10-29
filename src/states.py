"""Training states: roll-ins, frozen-encoder negative pools, and the gold/visited mix of each training condition.

Conditions used in the paper:
  RN              teacher forcing: gold states only (random gold subsets in chain order)
  OP-refresh      on-policy state training: half gold states, half visited states re-collected every epoch
  Fixed-OP        w/o refresh: the visited half is sampled once from one teacher-forced roll-in and kept
  Pool-resample   the visited half is redrawn every epoch from that same teacher-forced roll-in
  Static-corrupt  synthetic contamination of gold states, built once
  Static-refresh  synthetic contamination of gold states, resampled every epoch
  Mix-25, Mix-75  25% / 75% visited states, refreshed every epoch (mixing ratio, Appendix C)
"""
from core import *
from collections import Counter

def wrong(v):return any(d not in v['gold'] for d in v['evidence'])
def statistics(rows):
 keys=[(v['qid'],tuple(v['evidence'])) for v in rows]
 return dict(states=len(rows),wrong_states=sum(wrong(v) for v in rows),wrong_state_fraction=sum(wrong(v) for v in rows)/len(rows),unique_states=len(set(keys)),duplicate_fraction=1-len(set(keys))/len(rows),steps=dict(Counter(v['node']+1 for v in rows)),evidence_passages=sum(len(v['evidence']) for v in rows),wrong_evidence_passages=sum(d not in v['gold'] for v in rows for d in v['evidence']),evidence_wrong_fraction=sum(d not in v['gold'] for v in rows for d in v['evidence'])/max(1,sum(len(v['evidence']) for v in rows)))
def rollin(enc,out):
 """Greedy T=4 roll-in of the current encoder on the training questions; every state that still misses a gold
 passage is labeled with the earliest missing one. Duplicate (question, history) states are removed."""
 out=Path(out);recs=read(ROOT/'inputs/train/questions.json');corp=corpus('train');D=torch.as_tensor(np.load(ROOT/'inputs/train/doc_emb.npy'),device='cpu');gold={v['qid']:v['gold'] for v in read(state_source('RN'))};rows=[];traces=[];tick=time.monotonic();enc.model.eval()
 with torch.inference_mode():
  for s in range(0,len(recs),32):
   jobs=[dict(rec=r,written=[],states=[],trajectory=0) for r in recs[s:s+32]]
   for t in range(4):
    texts=[text19(j['rec']['question'],j['written'],corp) for j in jobs]
    qs=np.concatenate([enc.encode(texts[a:a+32]) for a in range(0,len(texts),32)])
    for j,text,q in zip(jobs,texts,qs):
     key=j['rec']['qid'];e=j['written'].copy();remaining=[d for d in gold[key] if d not in e]
     if remaining:j['states'].append(dict(qid=key,node=t,evidence=e,gold=gold[key],remaining=remaining,positives=remaining[:1],text=text))
     scores=(torch.as_tensor(np.asarray([np.zeros_like(q),q]))@D.T).numpy()[1].copy();scores[e]=-np.inf;ids=np.flatnonzero(np.isfinite(scores));order=ids[np.lexsort((ids,-scores[ids]))]
     choice=int(order[0])
     j['written'].append(choice)
   values=[dict(qid=j['rec']['qid'],trajectory=j['trajectory'],writes=j['written']) for j in jobs];save(out/'rollin'/f'{s:06d}.json',values);traces+=values;rows.extend(v for j in jobs for v in j['states']);save(out/'status.json',dict(stage='current model rollin',completed=min(s+32,len(recs)),total=len(recs)));print('rollin',s,flush=True)
 seen=set();unique=[]
 for v in rows:
  key=(v['qid'],tuple(v['evidence']))
  if key not in seen:unique.append(v);seen.add(key)
 save(out/'rollin_traces.json',traces);save(out/'rollin_stats.json',dict(training_questions=len(recs),trajectories=len(traces),states_before_dedup=len(rows),states_after_dedup=len(unique),dedup_removed=len(rows)-len(unique),rollin_seconds=time.monotonic()-tick,split='train',**statistics(unique)))
 return unique

def pools(enc,rows,out):
 """Negative pool of every state: top 100 of the frozen encoder (LoRA disabled) for x(s), gold passages removed."""
 D=torch.as_tensor(np.load(ROOT/'inputs/train/doc_emb.npy'),device='cpu');order=np.arange(len(D));tick=time.monotonic();enc.model.eval()
 with enc.model.disable_adapter(),torch.inference_mode():
  for s in range(0,len(rows),32):
   qs=enc.encode([v['text'] for v in rows[s:s+32]])
   for v,q in zip(rows[s:s+32],qs):
    scores=(torch.as_tensor(q)@D.T).numpy();rank=np.lexsort((order,-scores));v['negative_top100']=[int(d) for d in rank if int(d) not in set(v['gold'])][:100]
   save(Path(out)/'status.json',dict(stage='frozen negative pools',completed=min(s+32,len(rows)),total=len(rows)));print('pools',s,flush=True)
 save(Path(out)/'pool_cost.json',dict(seconds=time.monotonic()-tick,view='base encoder with adapters disabled; frozen eval; top100 non-gold'))
 return rows

def refreshed_mix(c,seed,epoch,pool=None):
 """OP-refresh: the gold half is fixed; the visited half is the epoch-1 teacher-forced roll-in, then a uniform
 sample of the current encoder's roll-in at every later epoch."""
 original=reference();valid=valid_qids();fixed=[v for v in original if v['qid'] not in valid and v['origin']=='R17-RN'];policy=[v for v in original if v['qid'] not in valid and v['origin']=='RN-on-policy'];validation=[v for v in original if v['qid'] in valid];rng=random.Random(seed+1000*epoch)
 if epoch==0:selected=policy;fit=[v for v in original if v['qid'] not in valid]
 else:
  available=[v for v in pool if v['qid'] not in valid];selected=rng.sample(available,len(policy));fit=fixed+[dict(v,origin='OP-refresh') for v in selected];rng.shuffle(fit)
 out=ROOT/'states'/c/str(seed)/f'epoch_{epoch+1:02d}';save(out/'op_reference_states.json',fit+validation)
 stats=dict(condition=c,seed=seed,epoch_number=epoch+1,fit_states=len(fit),gold_half=len(fixed),policy_half=len(selected),validation_states=len(validation),**statistics(selected));save(out/'state_stats.json',stats);save(out/'states.json',fit+validation)
 return fit+validation,stats

def make_mix(c,seed,epoch,pool=None):
 if c=='OP-refresh':return refreshed_mix(c,seed,epoch,pool)
 valid=valid_qids();rn=read(state_source('RN'));fit_rn=[v for v in rn if v['qid'] not in valid];validation=[v for v in rn if v['qid'] in valid];rng=random.Random(seed+1000*epoch)
 out=ROOT/'states'/c/str(seed)/f'epoch_{epoch+1:02d}'
 if c=='RN':
  rows=rn;fit=[v for v in rows if v['qid'] not in valid];save(out/'states.json',rows);return rows,dict(fit_states=len(fit),validation_states=len(rows)-len(fit))
 original=reference();fixed=[v for v in original if v['qid'] not in valid and v['origin']=='R17-RN'];policy=[v for v in original if v['qid'] not in valid and v['origin']=='RN-on-policy'];n=len(fit_rn);np_=len(policy)
 if c in ['Static-corrupt','Static-refresh','Fixed-OP','Pool-resample']:fixed=random.Random(seed).sample(fixed,n//2);np_=n-len(fixed)
 if c in ['Static-corrupt','Static-refresh']:
  dest=ROOT/'states'/c/str(seed)/('frozen_states.json' if c=='Static-corrupt' else f'epoch_{epoch+1:02d}/states.json')
  if dest.exists():rows=read(dest);return rows,read(dest.with_name('state_stats.json'))
  empirical=read(ROOT/'static_reference'/f'{seed}.json');pairs=empirical['joint_samples'];by_g={g:[v for v in fit_rn if sum(d in v['gold'] for d in v['evidence'])==g] for g in range(4)}
  corp=corpus('train');questions={r['qid']:r['question'] for r in read(ROOT/'inputs/train/questions.json')};selected=[];mapping=[]
  # Draw a (wrong count, gold count) pair from the visited states, then a gold state with that gold count, and
  # insert that many passages of its frozen top 100 at random positions.
  for i in range(np_):
   m,g=rng.choice(pairs);v=rng.choice(by_g[g]);e=v['evidence'].copy();chosen=rng.sample([d for d in v['negative_top100'] if d not in e],m)
   for d in chosen:e.insert(rng.randrange(len(e)+1),d)
   u=dict(v,evidence=e,text=text19(questions[v['qid']],e,corp),origin=c);selected.append(u);mapping.append(dict(index=i,qid=v['qid'],m=m,gold_count=g,before=v['evidence'],after=e,pool=v['negative_top100']))
  fit=fixed+selected;rng.shuffle(fit);rows=fit+validation
  stats=dict(condition=c,seed=seed,fit_states=len(fit),validation_states=len(validation),model_rollouts=0,refreshes=0,reference=empirical['source'],reference_joint=empirical['joint_counts'],observed_joint=dict(Counter(str((v['m'],v['gold_count'])) for v in mapping)),**statistics(selected))
  save(dest,rows);save(dest.with_name('state_stats.json'),stats);save(dest.with_name('correspondence.json'),mapping)
 elif c=='Fixed-OP':
  dest=ROOT/'states'/c/str(seed)/'frozen_states.json'
  if dest.exists():rows=read(dest);return rows,read(dest.with_name('state_stats.json'))
  available=[v for v in read(ROOT/'states'/c/str(seed)/'pool/states.json') if v['qid'] not in valid];selected=rng.sample(available,np_);fit=fixed+[dict(v,origin=c) for v in selected];rng.shuffle(fit);rows=fit+validation;stats=dict(condition=c,seed=seed,fit_states=len(fit),generations=1,refreshes=0,**statistics(selected));save(dest,rows);save(dest.with_name('state_stats.json'),stats)
 elif c=='Pool-resample':
  # Same gold half and teacher-forced roll-in pool as Fixed-OP; the visited half is redrawn from that pool at every
  # epoch (epoch-specific rng), so epoch 1 draws exactly the Fixed-OP subset.
  available=[v for v in read(ROOT/'states'/'Fixed-OP'/str(seed)/'pool/states.json') if v['qid'] not in valid];selected=rng.sample(available,np_);fit=fixed+[dict(v,origin=c) for v in selected];rng.shuffle(fit);rows=fit+validation
  stats=dict(condition=c,seed=seed,epoch_number=epoch+1,fit_states=len(fit),validation_states=len(validation),policy_states=len(selected),pool_states=len(available),**statistics(selected));save(out/'states.json',rows);save(out/'state_stats.json',stats);return rows,stats
 else:
  # Mix-25 / Mix-75: the visited share is drawn from the teacher-forced roll-in in epoch 1, then from the current roll-in.
  available=[v for v in read(ROOT/'states/OP/onpolicy_states.json') if v['qid'] not in valid] if epoch==0 else [v for v in pool if v['qid'] not in valid]
  count=round(n*(.25 if c=='Mix-25' else .75));selected=rng.sample(available,count);clean=rng.sample(fit_rn,n-count);fit=clean+[dict(v,origin=c) for v in selected]
  rng.shuffle(fit);rows=fit+validation;stats=dict(condition=c,seed=seed,epoch_number=epoch+1,fit_states=len(fit),validation_states=len(validation),policy_states=len(selected),**statistics(selected))
 save(out/'states.json',rows);save(out/'state_stats.json',stats);return rows,stats
