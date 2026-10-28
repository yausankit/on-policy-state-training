"""R68: R30 official-MDR two-hop evaluator (beam 5, exact IP, additive chain score, all 25 chains),
generalised to an arbitrary corpus/questions directory. Model loading and encoding imported unchanged from round30/core30.py."""
import sys,os,time,argparse,gc
sys.path.insert(0,'/home/amax/revision_20260928/round30')
from core30 import *
import faiss
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--questions',default='questions.json');p.add_argument('--threads',type=int,default=8);a=p.parse_args()
torch.set_num_threads(a.threads);faiss.omp_set_num_threads(a.threads)
inp=Path(a.inputs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'complete.json').exists():raise SystemExit(0)
began=time.monotonic();corp=read(inp/'corpus.json');qs_all=read(inp/a.questions);tok=tokenizer();docfile=out/'doc_emb_mdr.npy'
if not docfile.exists():
 m=model('doc');chunks=[]
 for start in range(0,len(corp),1000):
  part=out/'doc_parts'/f'{start:07d}.npy';part.parent.mkdir(exist_ok=True)
  if not part.exists():np.save(part,encode(m,tok,[docpair(v) for v in corp[start:start+1000]],300,'doc'))
  chunks.append(np.load(part))
  if start%20000==0:print('doc',start,len(corp),time.strftime('%H:%M:%S'),flush=True)
 np.save(docfile,np.concatenate(chunks));del m,chunks;gc.collect();torch.cuda.empty_cache()
 import shutil;shutil.rmtree(out/'doc_parts')
D=np.load(docfile).astype('float32');index=faiss.IndexFlatIP(768);index.add(D);del D;m=model('q');rows=[]
for start in range(0,len(qs_all),100):
 part=out/'parts'/f'{start:04d}.json'
 if part.exists():values=read(part)
 else:
  batch=qs_all[start:start+100];qs=[question(v['question']) for v in batch];q=encode(m,tok,qs,70,'q');d1,i1=index.search(q,5);pairs=[]
  for bi in range(len(batch)):
   for j,docid in enumerate(i1[bi]):
    text=corp[docid]['text']
    if not text.strip():text=corp[docid]['title'];d1[bi,j]=-np.inf
    pairs.append((qs[bi],text))
  q2=encode(m,tok,pairs,350,'q');d2,i2=index.search(q2,5);d2=d2.reshape(len(batch),5,5);i2=i2.reshape(len(batch),5,5);scores=d1[:,:,None]+d2;values=[]
  for bi,rec in enumerate(batch):
   positions=np.vstack(np.unravel_index(np.argsort(scores[bi].ravel())[::-1],(5,5))).T
   chains=[[int(i1[bi,j]),int(i2[bi,j,k])] for j,k in positions];ranking=list(dict.fromkeys(d for chain in chains for d in chain))
   values.append(dict(qid=qid(rec),chains=chains,chain_scores=[float(scores[bi,j,k]) for j,k in positions],ranking=ranking))
  save(part,values)
 rows+=values;print('query',len(rows),len(qs_all),time.strftime('%H:%M:%S'),flush=True)
save(out/'chains.json',{v['qid']:dict(chains=v['chains'],scores=v['chain_scores']) for v in rows});save(out/'ranking_ids.json',{v['qid']:v['ranking'] for v in rows})
save(out/'complete.json',dict(questions=len(qs_all),passages=len(corp),seconds=time.monotonic()-began,gpu=torch.cuda.get_device_name(0),beam=5,hops=2,chains=25,max_q_len=70,max_c_len=300,max_q_sp_len=350,query_batch_size=100,encoding_microbatch=32,precision='torch autocast fp16, parameters fp32; float32 IP',chain_score='hop1 IP + hop2 IP',search='faiss IndexFlatIP CPU',official_classes=True,code='round30/core30.py unchanged + evaluate30 logic'))
