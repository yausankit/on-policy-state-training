"""Encoder, state serialization, chain readout and data paths shared by all scripts.

Query encoder: Qwen3-Embedding-0.6B with query-side LoRA (r=16, alpha=32, dropout 0.05 on q/k/v/o), last-token
pooling, L2 normalization. Documents are encoded once by the frozen model (inputs/<split>/doc_emb.npy).

Data layout under ROOT (default: this directory, override with OPST_ROOT):
  inputs/<split>/{questions.json,corpus.json,doc_emb.npy}    split in {train, design}; design = MuSiQue test questions
  states/RN/states.json                                      teacher-forced gold states with frozen top-100 pools
  states/OP/{states.json,onpolicy_states.json}               epoch-1 mix and the teacher-forced roll-in (initial_mix.py)
  validation_qids.json                                       the 200 checkpoint-selection questions
Set OPST_SHARED_NEG=1 for the shared-negative-pool control (Appendix B).
"""
import os,json,time,random,gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from peft import LoraConfig,get_peft_model,PeftModel
ROOT=Path(os.environ.get('OPST_ROOT',Path(__file__).resolve().parent))
DATASET=os.environ.get('R29_DATASET','musique')
SHARED_NEG=os.environ.get('OPST_SHARED_NEG','0')=='1'
QUERY_PROMPT='Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: '
INSTRUCTION='Instruct: Given a multi-hop retrieval step, produce an embedding for finding passages that can support the current subquery. Use the evidence state as context for placeholder variables, but do not replace, guess, or answer placeholder tokens. Emphasize entities, relations, temporal/location constraints, and evidence needed for the next hop.'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def progress(s):
 with (ROOT/'PROGRESS.md').open('a') as f:f.write('- '+time.strftime('%F %T')+' '+s+'\n')
def norm(t):return ' '.join(str(t or '').split())
def qid(r):return str(r.get('_id') or r.get('id') or r.get('uid') or r.get('qid'))
def corpus(split):
 return [norm(v.get('title'))+'\n'+norm(v.get('text') or v.get('paragraph_text') or '') for v in read(ROOT/'inputs'/split/'corpus.json')]
def text19(question,evidence,corp):
 """State text x(s): instruction, question, and every passage of the history in retrieval order."""
 passages=[corp[d] for d in evidence];chars=max([len(p) for p in passages]+[1])
 state='Evidence state:\n'+'\n'.join(f'[{i+1}] {norm(d)[:chars]}' for i,d in enumerate(passages)) if passages else 'Evidence state: <EMPTY>'
 return INSTRUCTION+'\nCurrent subquery: '+norm('Original question: '+question)+'\n'+state

def readout(written,aggregate):
 """Ranking: the chain's writes first, then the remaining passages by their best score over all steps."""
 prefix=[d for i in sorted(written) for d in written[i]];seen=set(prefix);order=np.lexsort((np.arange(len(aggregate)),-aggregate))
 return prefix+[int(d) for d in order if d not in seen][:max(0,50-len(prefix))]
def adapter(c,seed=13):return ROOT/'lora'/c/str(seed)/'adapter'
def state_source(c):return ROOT/'states/RN/states.json' if c=='RN' else ROOT/'states'/c/'states.json'
def outpath(c,split,seed=13,gpu=None):return ROOT/'evaluations'/c/str(seed)/split/(gpu or os.environ.get('R19_GPU','gpu'))
def setup(seed=13):
 torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def modelpath():return os.environ.get('R19_MODEL','Qwen/Qwen3-Embedding-0.6B')
class Encoder:
 def __init__(self,path=None,training=False,frozen=False):
  self.tokenizer=AutoTokenizer.from_pretrained(modelpath(),padding_side='left',local_files_only=True)
  base=AutoModel.from_pretrained(modelpath(),torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
  self.model=base if frozen else get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none')) if path is None else PeftModel.from_pretrained(base,str(path),is_trainable=training)
  if training:
   if os.environ.get('R29_CHECKPOINTING','0')=='1':self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});self.model.enable_input_require_grads()
   self.model.config.use_cache=False
  else:self.model.eval()
 def vector(self,texts,train=False):
  toks=self.tokenizer([QUERY_PROMPT+norm(t) for t in texts],padding=True,truncation=True,max_length=32768,return_tensors='pt').to('cuda')
  with torch.autocast('cuda',dtype=torch.float16):h=self.model(**toks).last_hidden_state[:,-1]
  return torch.nn.functional.normalize(h.float() if train else h,p=2,dim=1)
 def encode(self,texts,fp32=False):
  clean=[norm(t) for t in texts];order=sorted(range(len(clean)),key=lambda i:len(clean[i]));formatted=[clean[i] for i in order];result=[]
  with torch.no_grad():
   for start in range(0,len(formatted),2):result.append(self.vector(formatted[start:start+2],train=fp32).float().cpu().numpy())
  sorted_arr=np.concatenate(result);arr=np.empty_like(sorted_arr);arr[np.asarray(order,dtype=np.int64)]=sorted_arr
  return arr
 def release(self):self.model=None;gc.collect();torch.cuda.empty_cache()
def device_info():return dict(host=os.environ.get('R8_HOST',''),bus=os.environ.get('R8_BUS',''),gpu=os.environ.get('R19_GPU',''),name=torch.cuda.get_device_name(0),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

def valid_qids():return set(read(ROOT/'validation_qids.json')['qids'])
def reference():return read(ROOT/'states/OP/states.json')
# Shared negative pools (Appendix B): one frozen question-only top-100 non-gold pool per question, used for every
# state of that question in place of its per-state pool.
SHARED_POOL=ROOT/'shared_pools/train_question_top100.json'
def apply_shared(rows,shared):return [dict(v,negative_top100_state=v.get('negative_top100_state',v['negative_top100']),negative_top100=shared[v['qid']]) for v in rows]
