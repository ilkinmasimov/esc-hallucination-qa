# ESC Hallucination QA

**Mitigating Hallucinations in Medical Question Answering on ESC Cardiology Guidelines: Evaluation and Retrieval Optimization**

This repository contains the code, data artefacts, and frozen experimental results accompanying the master's thesis on detecting and correcting hallucinations in guideline-grounded medical question answering, built at Harbin Institute of Technology (Shenzhen).

## Overview

Large language models produce plausible but clinically wrong statements — wrong thresholds, wrong recommendation classes, wrong drug doses. This project studies whether a **verify-and-revise pipeline** can catch and fix such errors when the reference standard is a current clinical guideline, and — more importantly — **measures how reliable that pipeline itself is**, claim by claim, against human judgement.

Four recent ESC guidelines are covered:

| Guideline | Year | Code |
|-----------|------|------|
| ESC/EACTS Valvular Heart Disease | 2025 | VHD |
| ESC Atrial Fibrillation | 2024 | AF |
| ESC Cardiomyopathies | 2023 | CM |
| ESC CVD Prevention | 2021 | CVD |

## Pipeline

Five stages, verify-and-revise style:

1. **Generation** — initial answer from the question alone, no retrieval (GPT-5.4)
2. **Decomposition** — answer split into atomic claims, typed `guideline_fact` / `patient_specific` (GPT-4.1-mini)
3. **Retrieval** — top-5 chunks per claim, FAISS flat inner-product index, `bge-small-en-v1.5` embeddings
4. **Verification** — each claim judged *supported / refuted / unverifiable* with cited chunk and rationale (GPT-4.1-mini)
5. **Correction** — refuted claims rewritten, up to 3 iterations (GPT-5.4)

Four retrieval arms are compared on shared inputs: `coarse` (525 chunks), `atomic` (1035 chunks), `atomic + guideline filter`, `atomic + guideline + section filter`. Each arm runs 3 times over all 160 questions: **1,920 arm-runs** in total.

## Dataset

160 expert-reviewed questions (100 patient vignettes grounded in 89 cases from 82 published case reports, 60 fact-based), all approved by a practising internist in two documented review rounds. The retrieval corpus comprises 627 hand-verified structured guideline records.

- `dataset/dataset_reviewed.jsonl` — the locked dataset
- `dataset/construction/case_info_all.jsonl` — question-to-case-report mapping (raw extracts are not redistributed)

## Headline results

| Quantity | Value |
|---|---|
| First-pass flagged instances audited (by hand) | 606 |
| True hallucinations (instances / rows) | 284 / 59 |
| Verifier true-hallucination precision across arms | 44.7%–50.6% |
| Single-run TH recall → 3-run union recall | 35.6–41.2% → 71–80% |
| Genuine-error final answers | 32 of 1,920 (1.7%) |
| Correction-introduced share of shipped genuine errors | 35% (17 of 49) |
| Verifier–judge agreement (Cohen's κ) | 74.2% / 0.505 |
| Accepted-claim miss-rate (blind expert probe, stratified) | 3.2% (95% CI 0.7–6.7%) |
| Total project API cost | $51.82 |

## Repository structure

```
dataset/                  locked dataset + construction metadata
experiments/
  config.py               paths and constants (models, arms, top-k, iterations)
  build_corpus.py         corpus and FAISS index builder (coarse + atomic)
  pipeline_final.ipynb    experiment runs (notebook of record)
  analysis/               analysis.ipynb, freeze manifest, all citable CSV tables
  audit/                  audit_claims.csv (160 audited rows), audit.ipynb
  rag_index/              chunk metadata + build report (rebuild indexes with build_corpus.py)
  results/                15 frozen run payloads + combined files
  results/step7_judgments.jsonl   906 second-judge verdicts
thesis/                   LaTeX source of the thesis
visuals/                  figure sources
```

## Reproducibility

All artefacts are frozen as of 2026-11-04 with SHA-256 hashes in `experiments/analysis/freeze_manifest.csv`. Reconciliation checks at freeze: 15 run payloads, 906 judge judgements, 160 dataset rows, 59 confirmed true-hallucination rows. See `experiments/analysis/FREEZE.md` for the environment and notebooks of record.

Pipeline run order: `build_corpus.py` → `pipeline_final.ipynb` → `analysis/analysis.ipynb`.

## Cost

$35.42 pipeline (OpenAI) + $6.34 second-verifier study + $10.06 dataset construction (Qwen3-Max) = **$51.82 total** (provider billing records; see `experiments/analysis/step6_cost.csv`).

## Citation

If you use this dataset, pipeline, or audit methodology, please cite the thesis:

```
Ilkin Masimov. Mitigating Hallucinations in Medical Question Answering on ESC
Cardiology Guidelines: Evaluation and Retrieval Optimization. Master's thesis,
Harbin Institute of Technology, Shenzhen, 2026.
```
