"""Sanity: R68 MuSiQue ranking vs R30 ranking, and R@5/R@10/C@5/C@10 by R30 text-matching gold."""
import json,sys
R30='/home/amax/revision_20260928/round30/musique/inputs/official/';W=sys.argv[1]
key=lambda t:''.join(t.split())
raw=json.load(open(R30+'raw.json'));corp=json.load(open(R30+'corpus.json'));texts=[key(v['title']+'\n'+v['text']) for v in corp]
rk=json.load(open(W+'/ranking_ids.json'));old=json.load(open(sys.argv[2]))
same=sum(rk[q][:10]==old[q][:10] for q in rk);M={}
for k in (5,10):
 R=[];C=[]
 for rec in raw:
  q=str(rec.get('_id') or rec.get('id'));gold={key(v['title']+'\n'+v['paragraph_text']) for v in rec['paragraphs'] if v['is_supporting']};h=len(gold&{texts[d] for d in rk[q][:k]});R.append(h/len(gold));C.append(h==len(gold))
 M[f'R@{k}']=round(100*sum(R)/len(R),1);M[f'C@{k}']=round(100*sum(C)/len(C),1)
print(json.dumps(dict(n=len(rk),identical_top10_vs_r30=same,**M)))
