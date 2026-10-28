import sys,os,numpy as np,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from run69 import QwenST
from common69 import load_corpus, load_questions, DATA
sys.path.insert(0,'/home/amax/revision_20260928/round29'); 
m=QwenST(); c=load_corpus('musique'); d=np.load(DATA['musique']['dir']+'/doc_emb.npy',mmap_mode='r')
e=m.encode(c[:64]); print('doc cos', (e*np.asarray(d[:64],np.float32)).sum(1).min())
os.environ.setdefault('R19_MODEL','/home/amax/revision_20260928/round8/encoder/model')
import core29
ref=core29.Encoder(frozen=True); q=[x[1] for x in load_questions('musique')[:8]]
a=np.stack([m.encode(x) for x in q]); b=ref.encode(q); print('query cos', (a*b).sum(1).min())
