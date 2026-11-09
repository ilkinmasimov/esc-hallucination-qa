"""
config.py  -  single source of truth for paths + constants.
Imported by BOTH build_corpus.ipynb and pipeline.ipynb so paths live in one place.

Usage in a notebook cell:
    import sys; sys.path.insert(0, "/content/drive/MyDrive/medical_qa_hallucination/experiments")
    from config import *
"""
import os

# ----------------------------------------------------------------------------- roots
ROOT        = "/content/drive/MyDrive/medical_qa_hallucination"
DATASET_DIR = os.path.join(ROOT, "dataset")
GUIDELINES  = os.path.join(DATASET_DIR, "guidelines")          # the 4 ESC folders live here
EXPERIMENTS = os.path.join(ROOT, "experiments")
INDEX_DIR   = os.path.join(EXPERIMENTS, "rag_index")           # faiss + meta outputs
RESULTS_DIR = os.path.join(EXPERIMENTS, "results")             # every run saved here
RESULTS_FINAL = os.path.join(RESULTS_DIR, "final")
os.makedirs(RESULTS_FINAL, exist_ok=True)


# the dataset the pipeline evaluates against.
# dataset_verified.jsonl = all 160 with expert-review columns
#   (expert_reviewed, expert_corrected, final_question_text).
# The pipeline selects the 20 expert-reviewed rows and reads final_question_text.

# round-1 frozen dataset (used by the archived reviewed20 notebook)
DATASET_REVIEWED20 = os.path.join(DATASET_DIR, "expert_review", "expert_pass", "dataset_verified.jsonl")
DATASET = DATASET_REVIEWED20   # alias: the archived pipeline.ipynb still runs unchanged

# final 160-question verified dataset (round 1 + round 2 merged)
DATASET_FINAL = os.path.join(DATASET_DIR, "expert_review", "expert_pass", "dataset_verified_final.jsonl")

# ----------------------------------------------------------------------------- corpus build
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # free, local; FIXED across all arms
# recursive glob, now rooted at the guidelines/ folder (new structure)
MASTER_GLOB = os.path.join(GUIDELINES, "**", "guideline_master.jsonl")
SKIP_PATH_SUBSTR = ["_old"]              # drop superseded/stale section folders

# ----------------------------------------------------------------------------- pipeline knobs
GEN_MODEL    = "gpt-5.4"          # initial answer + rewrites (strong)
VERIFY_MODEL = "gpt-4.1-mini"     # decompose + verify + recompose (cheap)
MAX_ITERS    = 3                  # rewrite-loop cap (termination guard)
TOP_K        = 5                  # chunks retrieved per claim
SEED         = 42

# the experiment arms (name -> (index, filter_mode))
#   filter_mode: None | "guideline" | "section_group"
ARMS = {
    "coarse":            ("coarse", None),            # baseline
    "atomic":            ("atomic", None),            # optimized chunking
    "atomic_gl_filter":  ("atomic", "guideline"),     # + restrict to guideline (code)
    "atomic_sec_filter": ("atomic", "section_group"), # + restrict to section group
}

# make sure output dirs exist on import
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
