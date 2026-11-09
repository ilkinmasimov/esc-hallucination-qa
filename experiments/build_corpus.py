"""
build_corpus.py  -  build the RAG retrieval corpus + indexes for the
hallucination-mitigation pipeline.   (v2: + section_group metadata)

Reads every guideline_master.jsonl under GUIDELINES (recursively) and builds
TWO retrieval indexes so the thesis can compare chunking strategies:

  * COARSE  (baseline) : one chunk per record
  * ATOMIC  (optimized): recommendation tables -> per-recommendation;
                         whitelisted lookup tables -> per-row; prose/figure/box whole.

Both strategies share IDENTICAL prose/figure/box handling; the contrast lives
entirely in how structured tables are split. That keeps the experiment clean.

NEW in v2: every chunk carries `section_group` (e.g. "CM_S6-10-6-11"), derived
from the folder the guideline_master.jsonl was read from. This is the key that
lets retrieval be filtered to the exact curated section a question came from
(question uid CM_S6-10-6-11_Q1 -> section_group CM_S6-10-6-11). Chunks also keep
`code` (CM/AF/VHD/CVD) for coarser guideline-level filtering.

Embeddings: a LOCAL sentence-transformers model (no API key). Indexes: FAISS.
Outputs to INDEX_DIR:
    coarse.faiss   coarse_meta.jsonl
    atomic.faiss   atomic_meta.jsonl
    build_report.txt

Run in Colab (after mounting Drive):
    !pip -q install sentence-transformers faiss-cpu
    import sys; sys.path.insert(0, "/content/drive/MyDrive/medical_qa_hallucination/experiments")
    from config import *
    %run /content/drive/MyDrive/medical_qa_hallucination/experiments/build_corpus.py
"""

import os, re, glob, json, sys

# ----------------------------------------------------------------------------- config
# Prefer values from config.py if it has been imported into the namespace;
# otherwise fall back to explicit defaults (so the file also runs standalone).
try:
    INDEX_DIR
    MASTER_GLOB
    EMBED_MODEL
    SKIP_PATH_SUBSTR
    GUIDELINES
except NameError:
    ROOT        = "/content/drive/MyDrive/medical_qa_hallucination"
    GUIDELINES  = os.path.join(ROOT, "dataset", "guidelines")
    INDEX_DIR   = os.path.join(ROOT, "experiments", "rag_index")
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"
    MASTER_GLOB = os.path.join(GUIDELINES, "**", "guideline_master.jsonl")
    SKIP_PATH_SUBSTR = ["_old"]

# generic boilerplate tables repeated in every guideline file -> not indexed,
# but their definitions enrich class/level wording on recommendation chunks.
SKIP_TABLE_IDS = {"Table_1", "Table_2"}

# ATOMIC-mode: non-recommendation tables to split into one chunk PER ROW.
SPLIT_TABLE_IDS = {
    ("AF",  "Table_5"),   ("AF",  "Table_6"),  ("AF",  "Table_7"),
    ("AF",  "Table_10"),  ("AF",  "Table_11"), ("AF",  "Table_13"),
    ("CM",  "Table_15"),  ("CM",  "Table_19"), ("CM",  "Table_21"),
    ("CM",  "Table_25"),
    ("CVD", "Table_5"),   ("CVD", "Table_12"), ("CVD", "Table_13"),
    ("CVD", "Table_18"),
}

# candidate fields used to label a split row (first match wins)
ROW_LABEL_KEYS = ("gene", "drug", "type", "concept", "score", "component",
                  "category", "condition", "risk_factor", "group", "name",
                  "intensity_level", "issue", "finding", "parameter",
                  "risk_level", "stage")

CLASS_WORDS = {
    "I":   "Class I (is recommended/indicated)",
    "IIa": "Class IIa (should be considered)",
    "IIb": "Class IIb (may be considered)",
    "III": "Class III (is not recommended)",
}
LEVEL_WORDS = {
    "A": "Level A (multiple RCTs or meta-analyses)",
    "B": "Level B (single RCT or large non-randomized studies)",
    "C": "Level C (expert consensus / small or retrospective studies)",
}

SOURCE_CODE = {
    "2025_ESC_VHD": "VHD",
    "2024_ESC_AF":  "AF",
    "2023_ESC_CM":  "CM",
    "2021_ESC_CVD": "CVD",
}

# map the guideline FOLDER name (e.g. "3_2023_ESC_CM") -> short code.
# used to derive section_group from the file path.
FOLDER_CODE = {
    "1_2025_ESC_VHD": "VHD",
    "2_2024_ESC_AF":  "AF",
    "3_2023_ESC_CM":  "CM",
    "4_2021_ESC_CVD": "CVD",
}

# ----------------------------------------------------------------------------- helpers
def norm_class(v):
    if v is None:
        return None
    s = re.sub(r"^Class\s+", "", str(v).strip(), flags=re.I)
    return s or None

def norm_level(v):
    if v is None:
        return None
    s = re.sub(r"^Level\s+", "", str(v).strip(), flags=re.I)
    return s or None

def code_of(source):
    return SOURCE_CODE.get(source, source or "UNK")

def clean(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


# ----------------------------------------------------------------------------- section_group derivation
def section_group_from_path(path):
    """
    Derive the section_group key from a guideline_master.jsonl path.

    Expected layout:
        .../guidelines/<GUIDELINE_FOLDER>/<SECTION_FOLDER>/guideline/guideline_master.jsonl
    e.g.
        .../guidelines/3_2023_ESC_CM/6-10-6-11/guideline/guideline_master.jsonl
        -> code "CM", section "6-10-6-11"  ->  "CM_S6-10-6-11"

    Returns (section_group, code).  Falls back gracefully if the layout differs.
    """
    parts = os.path.normpath(path).split(os.sep)
    # find the guideline folder (one whose name is in FOLDER_CODE)
    code, sec = None, None
    for i, p in enumerate(parts):
        if p in FOLDER_CODE:
            code = FOLDER_CODE[p]
            # the SECTION folder is the next path component after the guideline folder
            if i + 1 < len(parts):
                sec = parts[i + 1]
            break
    if code and sec:
        return f"{code}_S{sec}", code
    # fallback: unknown layout
    return "UNK", (code or "UNK")


# ----------------------------------------------------------------------------- flatten
def flatten(obj, prefix=""):
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            label = str(k).replace("_", " ")
            if isinstance(v, (dict, list)):
                sub = flatten(v, prefix + "  ")
                if sub:
                    lines.append(f"{prefix}{label}:")
                    lines.extend(sub)
            else:
                val = clean(v)
                if val:
                    lines.append(f"{prefix}{label}: {val}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.extend(flatten(item, prefix))
            else:
                val = clean(item)
                if val:
                    lines.append(f"{prefix}- {val}")
    else:
        val = clean(obj)
        if val:
            lines.append(f"{prefix}{val}")
    return lines

def flatten_text(obj):
    return "\n".join(flatten(obj)).strip()


# ----------------------------------------------------------------------------- recommendation extraction
REC_TEXT_KEYS = ("recommendation", "text", "statement", "definition")

def extract_recommendations(node, group=None, acc=None):
    if acc is None:
        acc = []
    if isinstance(node, dict):
        text_key = next((k for k in REC_TEXT_KEYS if k in node and isinstance(node[k], str)), None)
        has_cls  = "class" in node or "level" in node
        if text_key and has_cls:
            rec = {
                "text":  clean(node.get(text_key)),
                "class": norm_class(node.get("class")),
                "level": norm_level(node.get("level")),
                "note":  clean(node.get("note")),
                "group": group,
            }
            if rec["text"]:
                acc.append(rec)
            return acc
        for k, v in node.items():
            if k == "abbreviations":
                continue
            if isinstance(v, (dict, list)):
                new_group = group
                if isinstance(node.get("section"), str):
                    new_group = node["section"]
                elif isinstance(node.get("category"), str):
                    new_group = node["category"]
                else:
                    nk = str(k).replace("_", " ")
                    if nk not in ("recommendations", "recommendation groups",
                                  "pathway", "sections", "genes"):
                        new_group = nk if nk not in ("table",) else group
                extract_recommendations(v, new_group, acc)
    elif isinstance(node, list):
        for item in node:
            extract_recommendations(item, group, acc)
    return acc


# ----------------------------------------------------------------------------- record -> chunks
def section_label(rec):
    sid = rec.get("subsection_id") or rec.get("section_id")
    title = rec.get("subsection_title") or rec.get("section_title")
    if sid and title:
        return f"{sid} {clean(title)}"
    return clean(sid or title or "")

def header(code, section, extra=None):
    bits = [code]
    if section:
        bits.append(f"\u00a7{section}")
    if extra:
        bits.append(extra)
    return "[" + " \u00b7 ".join(bits) + "]"

def class_level_phrase(cls, lvl):
    bits = []
    if cls and cls in CLASS_WORDS:
        bits.append(CLASS_WORDS[cls])
    elif cls:
        bits.append(f"Class {cls}")
    if lvl and lvl in LEVEL_WORDS:
        bits.append(LEVEL_WORDS[lvl])
    elif lvl:
        bits.append(f"Level {lvl}")
    return " | ".join(bits)

def split_table_rows(tbl):
    if not isinstance(tbl, dict):
        return []
    for k, v in tbl.items():
        if k == "abbreviations":
            continue
        if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
            rows = []
            for item in v:
                label = ""
                for lk in ROW_LABEL_KEYS:
                    if lk in item and isinstance(item[lk], (str, int, float)):
                        label = clean(item[lk])
                        break
                rows.append((label, flatten_text(item)))
            return rows
    return []


def _base_meta(rec, code, section_group):
    """common metadata every chunk carries (incl. the new section_group)."""
    return {
        "source": rec.get("source"),
        "code": code,
        "section_group": section_group,   # <-- NEW: e.g. "CM_S6-10-6-11"
    }


def chunks_from_record(rec, mode, section_group):
    """Yield chunk dicts {text, meta} for one master record, per mode.
    `section_group` is derived from the file path and threaded in here."""
    code = code_of(rec.get("source"))
    ctype = rec.get("content_type")
    base = _base_meta(rec, code, section_group)
    out = []

    if ctype == "text":
        body = clean(rec.get("subsection_text"))
        if not body:
            return out
        sec = section_label(rec)
        head = header(code, sec)
        title = clean(rec.get("subsection_title"))
        text = f"{head}\n{title}\n{body}" if title else f"{head}\n{body}"
        out.append({"text": text, "meta": {
            **base, "content_type": "text", "section": sec, "chunk_type": "prose",
            "table_id": None, "figure_id": None, "class": None, "level": None,
        }})
        return out

    if ctype == "table":
        tid = rec.get("table_id")
        if tid in SKIP_TABLE_IDS:
            return out
        ttitle = clean(rec.get("table_title"))
        tbl = rec.get("table", {})

        if mode == "coarse":
            head = header(code, None, tid)
            out.append({"text": f"{head}\n{ttitle}\n{flatten_text(tbl)}", "meta": {
                **base, "content_type": "table", "section": None,
                "chunk_type": "table_whole",
                "table_id": tid, "figure_id": None, "class": None, "level": None,
            }})
            return out

        recs = extract_recommendations(tbl)
        if recs:
            for r in recs:
                cl = class_level_phrase(r["class"], r["level"])
                grp = f" \u00b7 {r['group']}" if r.get("group") else ""
                head = header(code, None, tid + (f"{grp}"))
                parts = [head, ttitle, r["text"]]
                if r.get("note"):
                    parts.append(f"Note: {r['note']}")
                if cl:
                    parts.append(cl)
                out.append({"text": "\n".join(p for p in parts if p), "meta": {
                    **base, "content_type": "table", "section": None,
                    "chunk_type": "recommendation",
                    "table_id": tid, "figure_id": None,
                    "class": r["class"], "level": r["level"],
                }})
        else:
            rows = split_table_rows(tbl) if (code, tid) in SPLIT_TABLE_IDS else []
            if rows:
                for label, row_text in rows:
                    extra = tid + (f" \u00b7 {label}" if label else "")
                    head = header(code, None, extra)
                    out.append({"text": f"{head}\n{ttitle}\n{row_text}", "meta": {
                        **base, "content_type": "table", "section": None,
                        "chunk_type": "table_row",
                        "table_id": tid, "figure_id": None, "class": None, "level": None,
                    }})
            else:
                head = header(code, None, tid)
                out.append({"text": f"{head}\n{ttitle}\n{flatten_text(tbl)}", "meta": {
                    **base, "content_type": "table", "section": None,
                    "chunk_type": "table_whole",
                    "table_id": tid, "figure_id": None, "class": None, "level": None,
                }})
        return out

    if ctype == "figure":
        fid = rec.get("figure_id")
        ftitle = clean(rec.get("figure_title"))
        fig = rec.get("figure", {})
        head = header(code, None, fid)
        out.append({"text": f"{head}\n{ftitle}\n{flatten_text(fig)}", "meta": {
            **base, "content_type": "figure", "section": None,
            "chunk_type": "figure_whole",
            "table_id": None, "figure_id": fid, "class": None, "level": None,
        }})
        return out

    if ctype == "box":
        bid = rec.get("box_id")
        btitle = clean(rec.get("box_title"))
        box = rec.get("box", {})
        head = header(code, None, bid)
        out.append({"text": f"{head}\n{btitle}\n{flatten_text(box)}", "meta": {
            **base, "content_type": "box", "section": None,
            "chunk_type": "box_whole",
            "table_id": None, "figure_id": None, "box_id": bid,
            "class": None, "level": None,
        }})
        return out

    return out


# ----------------------------------------------------------------------------- load
def load_records():
    """Return a list of (record, section_group) pairs + the file list.
    section_group is derived once per file from its path and attached to every
    record from that file."""
    files = sorted(glob.glob(MASTER_GLOB, recursive=True))
    files = [f for f in files if not any(s in f for s in SKIP_PATH_SUBSTR)]
    if not files:
        sys.exit(f"[FATAL] no guideline_master.jsonl found via {MASTER_GLOB}")
    print(f"Found {len(files)} guideline_master.jsonl files:")
    records = []                      # list of (rec, section_group)
    sg_counts = {}
    for f in files:
        sg, code = section_group_from_path(f)
        sg_counts[sg] = sg_counts.get(sg, 0)
        n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    import ast
                    rec = ast.literal_eval(line)
                records.append((rec, sg))
                n += 1
        sg_counts[sg] += n
        rel = os.path.relpath(f, GUIDELINES)
        print(f"   {rel:55s} sg={sg:18s} {n:4d} records")
    # sanity: warn about any file that did not resolve a section_group
    unk = [sg for sg in sg_counts if sg == "UNK"]
    if unk:
        print("\n[!] WARNING: some files resolved section_group='UNK' "
              "(check folder layout / FOLDER_CODE).")
    return records, files


def build_chunks(records, mode):
    chunks = []
    for rec, sg in records:
        chunks.extend(chunks_from_record(rec, mode, sg))
    return chunks


# ----------------------------------------------------------------------------- embed + index
def embed_and_index(chunks, name, model):
    import numpy as np, faiss
    texts = [c["text"] for c in chunks]
    print(f"  [{name}] embedding {len(texts)} chunks ...")
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{name}.faiss"))
    with open(os.path.join(INDEX_DIR, f"{name}_meta.jsonl"), "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({"text": c["text"], **c["meta"]}, ensure_ascii=False) + "\n")
    print(f"  [{name}] wrote {name}.faiss + {name}_meta.jsonl ({len(chunks)} chunks)")
    return index


# ----------------------------------------------------------------------------- search (test helper)
def load_index(name):
    import faiss
    index = faiss.read_index(os.path.join(INDEX_DIR, f"{name}.faiss"))
    meta = [json.loads(l) for l in open(os.path.join(INDEX_DIR, f"{name}_meta.jsonl"),
                                         encoding="utf-8") if l.strip()]
    return index, meta

def search(query, model, name="atomic", k=5, filters=None):
    """Return [(score, meta), ...]. If `filters` given (dict of meta_key->value),
    over-fetch then post-filter. Supports filtering on any meta field, including
    'code' (guideline) and 'section_group'."""
    import numpy as np
    index, meta = load_index(name)
    qv = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
    fetch = k * 20 if filters else k          # wider net when filtering (section groups are small)
    scores, idxs = index.search(qv, min(fetch, len(meta)))
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        m = meta[i]
        if filters and any(m.get(fk) != fv for fk, fv in filters.items()):
            continue
        results.append((float(score), m))
        if len(results) >= k:
            break
    return results


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    records, files = load_records()
    print(f"\nTotal records: {len(records)}")

    coarse = build_chunks(records, "coarse")
    atomic = build_chunks(records, "atomic")

    def summarize(chunks, label):
        by_code, by_type, by_sg = {}, {}, {}
        for c in chunks:
            m = c["meta"]
            by_code[m["code"]] = by_code.get(m["code"], 0) + 1
            by_type[m["chunk_type"]] = by_type.get(m["chunk_type"], 0) + 1
            by_sg[m["section_group"]] = by_sg.get(m["section_group"], 0) + 1
        print(f"\n[{label}] {len(chunks)} chunks")
        print("   by guideline:    ", dict(sorted(by_code.items())))
        print("   by type:         ", dict(sorted(by_type.items())))
        print(f"   section_groups:  {len(by_sg)} distinct")
        return by_sg

    summarize(coarse, "COARSE")
    atomic_sg = summarize(atomic, "ATOMIC")

    from sentence_transformers import SentenceTransformer
    print(f"\nloading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    embed_and_index(coarse, "coarse", model)
    embed_and_index(atomic, "atomic", model)

    # build report (now lists section_groups so you can eyeball uid alignment)
    with open(os.path.join(INDEX_DIR, "build_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"files: {len(files)}\nrecords: {len(records)}\n")
        fh.write(f"coarse chunks: {len(coarse)}\natomic chunks: {len(atomic)}\n")
        fh.write(f"embed model: {EMBED_MODEL}\n\n")
        fh.write("section_groups (atomic chunk counts):\n")
        for sg, n in sorted(atomic_sg.items()):
            fh.write(f"  {sg:20s} {n:4d}\n")

    # sample searches: plain vs section-group-filtered, to demo the filter
    print("\n--- sample searches (atomic, UNFILTERED) ---")
    for q in [
        "When is surgery indicated in asymptomatic severe aortic regurgitation?",
        "ICD primary prevention in non-dilated cardiomyopathy with LMNA",
    ]:
        print(f"\nQ: {q}")
        for score, m in search(q, model, name="atomic", k=3):
            loc = m.get("table_id") or m.get("figure_id") or m.get("section")
            print(f"  {score:.3f}  {m['code']:4s} {m['section_group']:16s} "
                  f"{m['chunk_type']:15s} {loc}")

    print("\n--- sample search (atomic, FILTERED to section_group CM_S7-3) ---")
    q = "ICD primary prevention in non-dilated cardiomyopathy with LMNA"
    print(f"Q: {q}")
    for score, m in search(q, model, name="atomic", k=3,
                           filters={"section_group": "CM_S7-3"}):
        loc = m.get("table_id") or m.get("figure_id") or m.get("section")
        print(f"  {score:.3f}  {m['code']:4s} {m['section_group']:16s} "
              f"{m['chunk_type']:15s} {loc}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
