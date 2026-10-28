"""Full effective batch16 and full in-batch negatives, query microbatch accumulation."""
from inference29 import *
from states29 import statistics
import shutil
import argparse
p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,default=13);p.add_argument('--microbatch',type=int,default=16);p.add_argument('--lr',type=float,required=True);cli=p.parse_args();c,seed,micro=cli.condition,cli.seed,cli.microbatch;method=c;os.environ['R29_CONDITION']=c;os.environ['R29_SEED']=str(seed)
out=ROOT/'lora'/c/str(seed);out.mkdir(parents=True,exist_ok=True)
if (out/'complete.json').exists():raise SystemExit(0)
setup(seed)
valid=valid_qids()
rows=read(ROOT/'attempts/sq_lora_v1_invalid/states/SQ-LoRA/states.json');initial_stats=dict(source='R26 frozen question-only top100 pools; one row per question')
save(ROOT/'states'/c/str(seed)/'states.json',rows)
texts=[r['text'] for r in rows];fit=[i for i,v in enumerate(rows) if v['qid'] not in valid];val=[i for i,v in enumerate(rows) if v['qid'] in valid]
save(ROOT/'states'/c/str(seed)/'training_state_stats.json',dict(condition=c,seed=seed,fit_states=len(fit),validation_states=len(val),**statistics([rows[i] for i in fit])))
began=time.monotonic();enc=Encoder(training=True);params=[p for p in enc.model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cli.lr,weight_decay=0);scaler=torch.amp.GradScaler('cuda');doc=torch.as_tensor(np.load(ROOT/'inputs/train/doc_emb.npy'),device='cuda');rng=random.Random(seed)
config=dict(seed=seed,r=16,alpha=32,dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],lr=cli.lr,weight_decay=0,epochs=10,patience=3,batch_states=16,forward_microbatch=micro,gradient_accumulation='query chunks over same full16-state candidate matrix; loss weighted by query count',temperature=.05,negatives_per_state=15,model_max_length=32768,fit_states=len(fit),validation_states=len(val),trainable_parameters=sum(p.numel() for p in params),state_source=str(ROOT/'states'/c/str(seed)),validation_selection='single-query C@5',device=device_info())
config.update(dataset=DATASET,objective='one uniformly sampled gold; other own gold masked including in-batch',unique_question_states=len({r['qid'] for r in rows}))
config.update(scheduler='linear warmup 5% then cosine to zero; fixed 10 epochs',gradient_checkpointing=os.environ.get('R29_CHECKPOINTING','0')=='1');save(out/'config.json',config);save(out/'validation_qids.json',dict(qids=sorted(valid)))
def candidates(ii,r):return np.asarray([[r.choice(rows[i]['positives']),*r.sample(rows[i]['negative_top100'],15)] for i in ii],np.int64)
vr=random.Random(13);validation=[(val[s:s+16],None) for s in range(0,len(val),16)];validation=[(ii,candidates(ii,vr)) for ii,_ in validation]
def batch(ii,ids,backward):
 d=torch.nn.functional.normalize(doc[torch.as_tensor(ids.reshape(-1),device='cuda')],dim=1);ce_total=0.;objective=0.
 for start in range(0,len(ii),micro):
  jj=ii[start:start+micro];q=enc.vector([texts[i] for i in jj],train=True);logits=q@d.T/.05;targets=torch.arange(start,start+len(jj),device='cuda')*16
  mask=np.asarray([[int(v) in set(rows[i]['remaining']) for v in ids.reshape(-1)] for i in jj],bool)
  mask[np.arange(len(jj)),np.arange(start,start+len(jj))*16]=False;mask=torch.as_tensor(mask,device='cuda');logits=logits.masked_fill(mask,-torch.inf)
  ce=torch.nn.functional.cross_entropy(logits,targets);value=ce
  weight=len(jj)/len(ii)
  if backward:scaler.scale(value*weight).backward()
  ce_total+=float(ce.detach())*weight;objective+=float(value.detach())*weight
 return ce_total,objective
validation_recs,validation_corp,validation_D,validation_gold=validation_inputs()
# Required R29-2 pretraining comparison using the same base weights, tokenizer and scorer.
enc.model.eval()
check_dir=ROOT/'checks'/'SQ-LoRA-v2'/str(seed)
with enc.model.disable_adapter():
 frozen_ranks=retrieve_single(enc,validation_recs,validation_corp,validation_D,check_dir/'frozen')
initial_ranks=retrieve_single(enc,validation_recs,validation_corp,validation_D,check_dir/'initial_adapter')
per_question={q:dict(frozen_C5=set(validation_gold[q]).issubset(frozen_ranks[q][:5]),initial_C5=set(validation_gold[q]).issubset(initial_ranks[q][:5]),same_top5=frozen_ranks[q][:5]==initial_ranks[q][:5]) for q in frozen_ranks}
check=dict(passed=all(v['frozen_C5']==v['initial_C5'] for v in per_question.values()),questions=len(per_question),frozen_C5=coverage(frozen_ranks,validation_gold),initial_C5=coverage(initial_ranks,validation_gold),per_question=per_question,base='same frozen encoder with adapter disabled; eval-mode, no optimizer steps')
save(check_dir/'pretraining_equivalence.json',check)
if not check['passed']:raise SystemExit('R29-2 pretraining single-query coverage differs from frozen encoder')

total_steps=10*((len(fit)+15)//16);warmup_steps=.05*total_steps;optimizer_step=0
def lr_at(step):return cli.lr*(step/warmup_steps if step<=warmup_steps else .5*(1+np.cos(np.pi*(step-warmup_steps)/(total_steps-warmup_steps))))
config.update(total_optimizer_steps=total_steps,warmup_steps=warmup_steps);save(out/'config.json',config)
curves=[];best=-1.;wait=3;selected=None;rollin_total=0.
for epoch in range(10):
 tick=time.monotonic();rollin_seconds=0.
 enc.model.train();order=fit.copy();rng.shuffle(order);losses=[]
 for s in range(0,len(order),16):
  ii=order[s:s+16];ids=candidates(ii,rng);opt.zero_grad(set_to_none=True);ce,obj=batch(ii,ids,True);optimizer_step+=1
  for group in opt.param_groups:group['lr']=lr_at(optimizer_step)
  scaler.step(opt);scaler.update();losses.append((ce,obj,len(ii)))
  if s%160==0:save(out/'status.json',dict(epoch=epoch,completed=min(s+16,len(order)),total=len(order),seconds=time.monotonic()-began));print(c,seed,epoch,s,ce,obj,flush=True)
 enc.model.eval();v=[]
 with torch.no_grad():
  for ii,ids in validation:v.append((*batch(ii,ids,False),len(ii)))
 vl=sum(x*n for x,_,n in v)/sum(n for _,_,n in v);tl=sum(x*n for x,_,n in losses)/sum(n for _,_,n in losses)
 checkpoint=out/'checkpoints'/f'epoch_{epoch+1:02d}';enc.model.save_pretrained(checkpoint)
 retrieval_tick=time.monotonic();ranks=retrieve_single(enc,validation_recs,validation_corp,validation_D,out/'validation'/f'epoch_{epoch+1:02d}');c5=coverage(ranks,validation_gold);retrieval_seconds=time.monotonic()-retrieval_tick
 if c5>best:best=c5;selected=epoch;wait=3
 else:wait-=1
 curves.append(dict(rollin_seconds=rollin_seconds,policy_state_stats=initial_stats,epoch=epoch,epoch_number=epoch+1,training_InfoNCE=tl,training_objective=sum(x*n for _,x,n in losses)/sum(n for _,_,n in losses),validation_InfoNCE=vl,validation_C5=c5,validation_retrieval_seconds=retrieval_seconds,selected_epoch=selected,seconds=time.monotonic()-tick));save(out/'training_curves.json',curves);progress(f'{c}/{seed} epoch={epoch+1} validation_C5={c5:.3f} InfoNCE={vl:.6f} selected={selected+1}');print(curves[-1],flush=True)
 if wait==0:break
shutil.copytree(out/'checkpoints'/f'epoch_{selected+1:02d}',out/'adapter')
for directory in (out/'checkpoints').iterdir():
 if directory.name not in [f'epoch_{selected+1:02d}',f'epoch_{epoch+1:02d}']:shutil.rmtree(directory)
summary=dict(selected_epoch=selected,selected_epoch_number=selected+1,best_validation_C5=best,early_stopped=wait==0,validation_retrieval_seconds=sum(v["validation_retrieval_seconds"] for v in curves),epochs=len(curves),rollin_seconds=rollin_total,gpu_seconds=time.monotonic()-began,**device_info());save(out/'complete.json',summary);(out/'summary.md').write_text(json.dumps(summary,indent=2)+'\n');progress(f'{c}/{seed} training complete')
