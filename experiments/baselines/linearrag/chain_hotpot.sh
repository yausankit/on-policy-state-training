while kill -0 3449866 2>/dev/null; do sleep 30; done
grep -q "^done" out/ner_hotpot_full.log || exit 1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 venv_ret/bin/python run69.py hotpot_full out/ner_hotpot_full.json out/rank_hotpot_full.json > out/run_hotpot_full.log 2>&1
