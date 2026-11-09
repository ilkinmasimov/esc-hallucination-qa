# Experiment freeze manifest

Freeze date: 2026-11-03
Python: 3.13.18

## Environment

- openai: 2.54.0
- pandas: 2.2.3
- numpy: 2.4.6
- scikit-learn: 1.6.1
- faiss-cpu: not installed
- sentence-transformers: 5.7.0
- transformers: 5.22.0
- torch: 2.11.0+cpu
- openpyxl: 3.1.5

## Frozen inputs (hashed in freeze_manifest.csv)

- dataset/: 1391 files, 307.7 MB
- experiments/results/: 35 files, 76.1 MB
- experiments/audit/: 9 files, 1.1 MB

## Reconciliation at freeze

- run payloads (results_*.json): 15 (expected 15)
- step7_judgments.jsonl lines: 906 (expected 906)
- dataset: combined/dataset_reviewed.jsonl (160 questions, locked)
- audit labels: post-2026-10-07 reconciliation (59 TH rows / 284 instances)

## Notebooks of record

- pipeline_final.ipynb (experiment runs)
- analysis.ipynb (Steps 1-6, cells B1-B6; Step 7 cells S7-1 to S7-9; robustness S7-8)
- qwen_usage.ipynb (dataset-generation cost accounting)