from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent
p=R/'inputs'
if not p.exists():p.symlink_to('../round25/inputs',target_is_directory=True)
(R/'checks').mkdir(exist_ok=True)
shutil.copyfile(R.parent/'round25/checks/validation_gold.json',R/'checks/validation_gold.json')
p=R/'states/SQ-LoRA';p.mkdir(parents=True,exist_ok=True)
shutil.copyfile(R.parent/'round26/musique/states/SQ-LoRA/states.json',p/'states.json')
for c in ['RN','OP-refresh','Noise-refresh','Gold-refresh']:
 for seed in [13,14,15]:
  old=R.parent/('round24' if seed==13 else 'round25') if c in ['RN','OP-refresh'] else R.parent/'round26/musique'
  oldc=c+'_3e-4' if c in ['RN','OP-refresh'] else c
  p=R/'references'/c/str(seed);src=old/'lora'/oldc/str(seed)
  if src.exists() and not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(src,target_is_directory=True)
# Fixed reference statistics are computed on 4070 and shipped as xz to every trainer.
