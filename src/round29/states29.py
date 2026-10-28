"""R29 train-only roll-ins, frozen-state negatives, exact R21 fixed gold half."""
from core29 import *
from collections import Counter

def wrong(v):return any(d not in v['gold'] for d in v['evidence'])
def statistics(rows):
 keys=[(v['qid'],tuple(v['evidence'])) for v in rows]
 return dict(states=len(rows),wrong_states=sum(wrong(v) for v in rows),wrong_state_fraction=sum(wrong(v) for v in rows)/len(rows),unique_states=len(set(keys)),duplicate_fraction=1-len(set(keys))/len(rows),steps=dict(Counter(v['node']+1 for v in rows)),evidence_passages=sum(len(v['evidence']) for v in rows),wrong_evidence_passages=sum(d not in v['gold'] for v in rows for d in v['evidence']),evidence_wrong_fraction=sum(d not in v['gold'] for v in rows for d in v['evidence'])/max(1,sum(len(v['evidence']) for v in rows)))
def rollin(enc,out,sample=False):
 out=Path(out);recs=read(ROOT/'inputs/train/questions.json');corp=corpus('train');D=torch.as_tensor(np.load(ROOT/'inputs/train/doc_emb.npy'),device='cpu');gold={v['qid']:v['gold'] for v in read(state_source('RN'))};rows=[];traces=[];rng=np.random.default_rng(13);tick=time.monotonic();enc.model.eval()
 with torch.inference_mode():
  for s in range(0,len(recs),32):
   jobs=[dict(rec=r,written=[],states=[],trajectory=k) for r in recs[s:s+32] for k in range(4 if sample else 1)]
   for t in range(4):
    texts=[text19(j['rec']['question'],j['written'],corp) for j in jobs]
    # Preserve R19 length sorting within each 32-query encoding chunk.
    qs=np.concatenate([enc.encode(texts[a:a+32]) for a in range(0,len(texts),32)])
    for j,text,q in zip(jobs,texts,qs):
     key=j['rec']['qid'];e=j['written'].copy();remaining=[d for d in gold[key] if d not in e]
     if remaining:j['states'].append(dict(qid=key,node=t,evidence=e,gold=gold[key],remaining=remaining,positives=remaining[:1],text=text))
     scores=(torch.as_tensor(np.asarray([np.zeros_like(q),q]))@D.T).numpy()[1].copy();scores[e]=-np.inf;ids=np.flatnonzero(np.isfinite(scores));order=ids[np.lexsort((ids,-scores[ids]))]
     if sample:
      top=order[:20];logits=scores[top].astype(np.float64)/.05;prob=np.exp(logits-logits.max());prob/=prob.sum();choice=int(rng.choice(top,p=prob))
     else:choice=int(order[0])
     j['written'].append(choice)
   values=[dict(qid=j['rec']['qid'],trajectory=j['trajectory'],writes=j['written']) for j in jobs];save(out/'rollin'/f'{s:06d}.json',values);traces+=values;rows.extend(v for j in jobs for v in j['states']);save(out/'status.json',dict(stage='sample rollin' if sample else 'current model rollin',completed=min(s+32,len(recs)),total=len(recs)));print('rollin',s,flush=True)
 seen=set();unique=[]
 for v in rows:
  key=(v['qid'],tuple(v['evidence']))
  if key not in seen:unique.append(v);seen.add(key)
 save(out/'rollin_traces.json',traces);save(out/'rollin_stats.json',dict(training_questions=len(recs),trajectories=len(traces),states_before_dedup=len(rows),states_after_dedup=len(unique),dedup_removed=len(rows)-len(unique),rollin_seconds=time.monotonic()-tick,split='train',corpus=str(ROOT/'inputs/train/corpus.json'),questions=str(ROOT/'inputs/train/questions.json'),sample=sample,topk=20 if sample else 1,temperature=.05 if sample else None,**statistics(unique)))
 return unique

def pools(enc,rows,out):
 D=torch.as_tensor(np.load(ROOT/'inputs/train/doc_emb.npy'),device='cpu');order=np.arange(len(D));tick=time.monotonic();enc.model.eval()
 # Same live base encoder, all LoRA adapters disabled: frozen R19 state view.
 with enc.model.disable_adapter(),torch.inference_mode():
  for s in range(0,len(rows),32):
   qs=enc.encode([v['text'] for v in rows[s:s+32]])
   for v,q in zip(rows[s:s+32],qs):
    scores=(torch.as_tensor(q)@D.T).numpy();rank=np.lexsort((order,-scores));v['negative_top100']=[int(d) for d in rank if int(d) not in set(v['gold'])][:100]
   save(Path(out)/'status.json',dict(stage='frozen negative pools',completed=min(s+32,len(rows)),total=len(rows)));print('pools',s,flush=True)
 save(Path(out)/'pool_cost.json',dict(seconds=time.monotonic()-tick,view='base encoder with adapters disabled; frozen eval; top100 non-gold'))
 return rows


def live_pools(enc,rows,out,epoch,checkpoint):
 D=torch.as_tensor(np.load(ROOT/'inputs/train/doc_emb.npy'),device='cpu');order=np.arange(len(D));tick=time.monotonic();enc.model.eval();output=[]
 with torch.inference_mode():
  for s in range(0,len(rows),32):
   batch=[dict(v) for v in rows[s:s+32]];qs=enc.encode([v['text'] for v in batch])
   for v,q in zip(batch,qs):
    scores=(torch.as_tensor(q)@D.T).numpy();rank=np.lexsort((order,-scores));v['negative_top100']=[int(d) for d in rank if int(d) not in set(v['gold'])][:100]
   output+=batch;save(Path(out)/'status.json',dict(stage='current-model negative refresh',completed=len(output),total=len(rows)));print('ANCE',s,flush=True)
 info=dict(seconds=time.monotonic()-tick,epoch_number=epoch+1,source_checkpoint=str(checkpoint),view='current live LoRA training encoder eval; adapters enabled',active_adapters=enc.model.active_adapters,optimizer_steps_this_epoch=0)
 save(Path(out)/'pool_cost.json',info);save(Path(out)/'states.json',output);save(Path(out)/'state_stats.json',dict(epoch_number=epoch+1,**statistics(output)))
 save(ROOT/'checks'/f'RN-ANCE_weights_{os.environ["R29_SEED"]}_{epoch+1}.json',dict(passed=True,**info,model_object_id=id(enc.model),rows=len(output),non_gold_pools=all(not set(v['gold'])&set(v['negative_top100']) for v in output),state_unchanged=all(all(a[k]==b[k] for k in ['qid','node','evidence','gold','remaining','positives','text']) for a,b in zip(rows,output))))
 return output,info

def legacy_make_mix(c,seed,epoch,pool=None):
 original=reference();valid=valid_qids();fixed=[v for v in original if v['qid'] not in valid and v['origin']=='R17-RN'];policy=[v for v in original if v['qid'] not in valid and v['origin']=='RN-on-policy'];validation=[v for v in original if v['qid'] in valid];rng=random.Random(seed+1000*epoch)
 if epoch==0:selected=policy;fit=[v for v in original if v['qid'] not in valid]
 else:
  available=[v for v in pool if v['qid'] not in valid];selected=rng.sample(available,len(policy));fit=fixed+[dict(v,origin='OP-refresh') for v in selected];rng.shuffle(fit)
 # Save the full OP-refresh control before applying the requested state intervention.
 out=ROOT/'states'/c/str(seed)/f'epoch_{epoch+1:02d}';save(out/'op_reference_states.json',fit+validation)
 if c in ['Noise-refresh','Gold-refresh']:
  transform_rng=random.Random(seed+1000*epoch+700000);selected=[v for v in fit if v['origin']!='R17-RN'];original_selected=selected;details=[]
  if c=='Noise-refresh':
   changed=[];corp=corpus('train');question={r['qid']:r['question'] for r in read(ROOT/'inputs/train/questions.json')}
   for idx,v in enumerate(selected):
    u=dict(v);e=v['evidence'];pool100=[d for d in v['negative_top100'] if d not in e or d not in v['gold']];used=set(d for d in e if d in v['gold']);new=[]
    for d in e:
     replacement=d if d in v['gold'] else transform_rng.choice([x for x in pool100 if x not in used]);new.append(replacement);used.add(replacement)
    u['evidence']=new;u['text']=text19(question[v['qid']],new,corp);changed.append(u);details.append(dict(index=idx,qid=v['qid'],node_before=v['node'],node_after=u['node'],before=e,after=new,pool=v['negative_top100']))
   selected=changed;passed=all(d['node_before']==d['node_after'] and len(d['before'])==len(d['after']) and [x for x in d['before'] if x in original_selected[i]['gold']]==[x for x in d['after'] if x in original_selected[i]['gold']] and all(b==a if a in original_selected[i]['gold'] else b in d['pool'] and b not in original_selected[i]['gold'] for a,b in zip(d['before'],d['after'])) for i,d in enumerate(details))
  else:
   good=[(i,v) for i,v in enumerate(selected) if not wrong(v)]
   full_pool=pool if epoch>0 else read(R19/'states/OP/onpolicy_states.json' if DATASET=='musique' else ROOT/'states/OP/onpolicy_states.json')
   if epoch==0:save(out/'raw_refresh_states.json',full_pool)
   available_good=[v for v in full_pool if v['qid'] not in valid and not wrong(v)]
   selected=[v for _,v in good]+transform_rng.choices(available_good,k=len(original_selected)-len(good));details=[dict(index=i,qid=v['qid'],evidence=v['evidence'],retained=True) for i,v in good]+[dict(qid=v['qid'],evidence=v['evidence'],retained=False) for v in selected[len(good):]];passed=all(not wrong(v) for v in selected) and all(v in available_good for v in selected[len(good):])
  # Preserve control shuffle positions for Noise; Gold fills those same policy positions.
  it=iter(selected);fit=[v if v['origin']=='R17-RN' else dict(next(it),origin=c) for v in fit]
  save(out/'correspondence.json',dict(passed=passed,condition=c,reference='op_reference_states.json',rows=details,duplicate_fraction=statistics(selected)['duplicate_fraction'],padding_pool='raw_refresh_states.json' if c=='Gold-refresh' and epoch==0 else 'refresh/states.json' if c=='Gold-refresh' else None))
 if c=='OP-RI':
  corp=corpus('train');questions={r['qid']:r['question'] for r in read(ROOT/'inputs/train/questions.json')};details=[]
  for v in fit:
   if v['origin']!='R17-RN' and wrong(v):
    clean=[d for d in v['evidence'] if d in v['gold']];v['positives']=v['remaining'].copy();v['clean_text']=text19(questions[v['qid']],clean,corp);details.append(dict(qid=v['qid'],evidence=v['evidence'],gold=v['gold'],clean_evidence=clean,original_text=v['text'],clean_text=v['clean_text']))
  save(out/'clean_state_check.json',dict(passed=all(v['clean_evidence']==[d for d in v['evidence'] if d in v['gold']] and v['original_text'].split('\nEvidence state:',1)[0]==v['clean_text'].split('\nEvidence state:',1)[0] for v in details),states=details))
 stats=dict(condition=c,seed=seed,epoch_number=epoch+1,fit_states=len(fit),gold_half=len(fixed),policy_half=len(selected),validation_states=len(validation),**statistics(selected));save(out/'state_stats.json',stats);save(out/'states.json',fit+validation)
 return fit+validation,stats

def make_mix(c,seed,epoch,pool=None):
 if c in ['OP-refresh','Noise-refresh']:return legacy_make_mix(c,seed,epoch,pool)
 valid=valid_qids();rn=read(state_source('RN'));fit_rn=[v for v in rn if v['qid'] not in valid];validation=[v for v in rn if v['qid'] in valid];rng=random.Random(seed+1000*epoch)
 out=ROOT/'states'/c/str(seed)/f'epoch_{epoch+1:02d}'
 if c in ['RN','SQ-LoRA']:
  sq={v['qid']:v for v in read(ROOT/'states/SQ-LoRA/states.json')} if c=='SQ-LoRA' else {};rows=rn if c=='RN' else [dict(sq[v['qid']],rn_source_index=i) for i,v in enumerate(rn)];fit=[v for v in rows if v['qid'] not in valid];save(out/'states.json',rows);return rows,dict(fit_states=len(fit),validation_states=len(rows)-len(fit))
 original=reference();fixed=[v for v in original if v['qid'] not in valid and v['origin']=='R17-RN'];policy=[v for v in original if v['qid'] not in valid and v['origin']=='RN-on-policy'];n=len(fit_rn);np_=len(policy)
 if c in ['Static-corrupt','Fixed-OP']:fixed=random.Random(seed).sample(fixed,n//2);np_=n-len(fixed)
 if c=='Static-corrupt':
  dest=ROOT/'states'/c/str(seed)/'frozen_states.json'
  if dest.exists():rows=read(dest);return rows,read(dest.with_name('state_stats.json'))
  empirical=read(ROOT/'static_reference'/f'{seed}.json');pairs=empirical['joint_samples'];by_g={g:[v for v in fit_rn if sum(d in v['gold'] for d in v['evidence'])==g] for g in range(4)}
  corp=corpus('train');questions={r['qid']:r['question'] for r in read(ROOT/'inputs/train/questions.json')};selected=[];mapping=[]
  # Resample joint (wrong count, gold count), then choose a clean RN state with that gold count.
  for i in range(np_):
   m,g=rng.choice(pairs);v=rng.choice(by_g[g]);e=v['evidence'].copy();chosen=rng.sample([d for d in v['negative_top100'] if d not in e],m)
   for d in chosen:e.insert(rng.randrange(len(e)+1),d)
   u=dict(v,evidence=e,text=text19(questions[v['qid']],e,corp),origin=c);selected.append(u);mapping.append(dict(index=i,qid=v['qid'],m=m,gold_count=g,before=v['evidence'],after=e,pool=v['negative_top100']))
  fit=fixed+selected;rng.shuffle(fit);rows=fit+validation
  stats=dict(condition=c,seed=seed,fit_states=len(fit),validation_states=len(validation),model_rollouts=0,refreshes=0,reference=empirical['source'],reference_joint=empirical['joint_counts'],observed_joint=dict(Counter(str((v['m'],v['gold_count'])) for v in mapping)),**statistics(selected))
  save(dest,rows);save(dest.with_name('state_stats.json'),stats);save(dest.with_name('correspondence.json'),mapping);save(ROOT/'checks'/f'Static-corrupt_{seed}.json',dict(passed=all(all(x in v['pool'] for x in v['after'] if x not in v['before']) for v in mapping),model_rollouts=0,refreshes=0,source='RN frozen state top100; CPU-only construction'))
 elif c=='Fixed-OP':
  dest=ROOT/'states'/c/str(seed)/'frozen_states.json'
  if dest.exists():rows=read(dest);return rows,read(dest.with_name('state_stats.json'))
  available=[v for v in read(ROOT/'states'/c/str(seed)/'pool/states.json') if v['qid'] not in valid];selected=rng.sample(available,np_);fit=fixed+[dict(v,origin=c) for v in selected];rng.shuffle(fit);rows=fit+validation;stats=dict(condition=c,seed=seed,fit_states=len(fit),generations=1,refreshes=0,**statistics(selected));save(dest,rows);save(dest.with_name('state_stats.json'),stats);save(ROOT/'checks'/f'Fixed-OP_{seed}.json',dict(passed=True,generations=1,refreshes=0,source=f'states/Fixed-OP/{seed}/pool/complete.json'))
 else:
  available=([v for v in read(R19/'states/OP/onpolicy_states.json') if v['qid'] not in valid] if c in ['Mix-25','Mix-75'] else policy) if epoch==0 else [v for v in pool if v['qid'] not in valid]
  if c=='Gold-refresh-matched':
   selected0=policy if epoch==0 else rng.sample(available,np_);selected=[dict(v,origin=c) for v in selected0 if not wrong(v)]
   # Immutable R17/R19 source row is the existing sample identity, as recorded by source_index.
   used={int(v['source_index']) for v in fixed};clean=[dict(v,rn_source_index=int(v['source_index'])) for v in fixed]
   reservoir=[(i,v) for i,v in enumerate(rn) if v['qid'] not in valid and i not in used]
   extra=rng.sample(reservoir,n-len(selected)-len(clean));clean += [dict(v,origin='R29-RN-fill',source_index=i,rn_source_index=i) for i,v in extra];fit=clean+selected
   save(out/'op_reference_states.json',selected0);save(out/'correspondence.json',dict(passed=len({v['rn_source_index'] for v in clean})==len(clean) and len(fit)==n and all(not wrong(v) for v in selected),identity='original immutable RN source row; R17 qid/node draw and R19 source_index; no RN row resampled',kept_policy=len(selected),rn_states=len(clean),rn_source_indices=[v['rn_source_index'] for v in clean],total=n,policy_padding=0,semantic_duplicate_fraction=statistics(fit)['duplicate_fraction']))
  else:
   count=round(n*(.25 if c=='Mix-25' else .75));selected=rng.sample(available,count);clean=rng.sample(fit_rn,n-count);fit=clean+[dict(v,origin=c) for v in selected]
  rng.shuffle(fit);rows=fit+validation;stats=dict(condition=c,seed=seed,epoch_number=epoch+1,fit_states=len(fit),validation_states=len(validation),policy_states=len(selected),**statistics(selected))
 save(out/'states.json',rows);save(out/'state_stats.json',stats);return rows,stats
