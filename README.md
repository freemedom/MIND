# 

论文里的Gemini-1.5-Flash代码里没有

# MIND
The official pytorch implement of - **MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection**.

## Repository layout

Top-level layout (conceptual; omitting `.git` and local editor metadata):

```
MIND/
├── SSR.py                 # Stage 1: Similar Sample Retrieval (CLIP)
├── RID.py                 # Stage 2: Relevant Insight Derivation (LLaVA)
├── IAI.py                 # Stage 3: Insight-Augmented Inference + metrics
├── requirements.txt       # Python dependencies (see note on LLaVA below)
├── LICENSE                # MIT License
├── utils/
│   ├── data_utils.py      # Per-dataset JSONL field mapping (FHM, HarM, MAMI)
│   ├── prompts.py         # Prompt templates for RID and IAI (debater / judge)
│   └── run_llava.py       # LLaVA model loading and inference helper
├── data/                  # Place downloaded datasets here (see Dataset)
│   └── README.md          # Same layout reminder as below
├── SSR/                   # Outputs: `{Dataset}_SSR.jsonl` (+ pre-computed examples)
│   └── README.md
├── RID/                   # Outputs: `{Dataset}_RID.jsonl` (+ pre-computed examples)
│   └── README.md
└── IAI/                   # Outputs: `{Dataset}_IAI.jsonl` (+ README)
    └── README.md
```

## What each part does

- **`SSR.py`** — Encodes test and train memes with OpenAI CLIP (`ViT-L/14@336px`), combines image and text features (weighted blend), cosine-similarity against the training pool, and writes the top-*k* similar training indices per test item to `SSR/{Dataset}_SSR.jsonl`.
- **`RID.py`** — Loads SSR lines and the training split; for each test item, walks the top similar training memes through LLaVA using `RID_prompt`, iteratively updating high-level “rules” in forward and reversed meme order; appends `forward` / `backward` rule strings to `RID/{Dataset}_RID.jsonl`. Supports resume by appending if the output file already exists.
- **`IAI.py`** — For each test meme, runs two debaters (same image, rules from `forward` vs `backward`); if they disagree, a judge prompt resolves the label. Writes per-item predictions and a final summary line with accuracy and macro-F1 to `IAI/{Dataset}_IAI.jsonl`. Resume logic skips already-written test rows and restores running accuracy state.
- **`utils/data_utils.py`** — Central `DATASET_CONFIGS` for FHM, HarM, and MAMI (image key, text key, label key, HarM list-label handling) and `get_item_data()` used by all stages.
- **`utils/prompts.py`** — String templates for RID rule induction and IAI debater/judge prompts.
- **`utils/run_llava.py`** — Wraps LLaVA’s `load_pretrained_model` and chat-style generation used by RID and IAI.

**Label convention:** evaluation uses binary labels (0 = harmless, 1 = harmful), aligned with `get_item_data()` for each dataset.

## SSR output format (`SSR/FHM_SSR.jsonl`)

The file is **JSON Lines**: one UTF-8 JSON object per line, no wrapping array. It is produced by `SSR.py` for the FHM dataset (same schema for `HarM_SSR.jsonl` and `MAMI_SSR.jsonl`).

Each line has three keys:

| Field | Type | Meaning |
|--------|------|---------|
| `index` | integer | Zero-based position of the **test** meme in `data/FHM/test.jsonl` (line order). |
| `samples` | list[int] | Length **10** (`k` in `process_clip_embeddings`). Each value is a zero-based index into **`data/FHM/train.jsonl`** (the training pool used for retrieval). |
| `scores` | list[float] | Same length as `samples`; cosine similarity between the test meme embedding and each retrieved train meme (CLIP image+text embedding, as implemented in `SSR.py`). Order matches `samples` (best match first). |

Example (one line, pretty-printed):

```json
{
  "index": 0,
  "samples": [7271, 5456, 7472, 105, 2918, 185, 4312, 8045, 7579, 4750],
  "scores": [0.7890625, 0.779296875, 0.77294921875, 0.7666015625, 0.76513671875, 0.74853515625, 0.7421875, 0.74072265625, 0.740234375, 0.73974609375]
}
```

**Coverage:** the shipped `FHM_SSR.jsonl` contains **500** lines, with `index` running from **0** through **499**—one retrieval record per FHM test example. Downstream **`RID.py`** uses the first few entries of `samples` (default 3) as analogs for rule induction.

## Dependencies

`requirements.txt` pins PyTorch ecosystem packages, CLIP (`openai-clip`), `transformers`, `sentence-transformers`, clustering/topic tooling (`hdbscan`, `umap-learn`, `bertopic`, `scikit-learn`, `scipy`), and utilities (`jsonlines`, `Pillow`, etc.).

**LLaVA:** `RID.py` and `IAI.py` import the `llava` package (e.g. `llava.model.builder`). It is **not** listed in `requirements.txt`; install the [LLaVA](https://github.com/haotian-liu/LLaVA) codebase in the same environment (or otherwise ensure `llava` is importable) before running those stages.

## Install

1. Clone the repo
```
git clone https://github.com/destroy-lonely/MIND.git
cd MIND
```

2. Install Package
```
conda create -n mind python=3.10 -y
conda activate mind
pip install -r requirements.txt
```

Install and configure LLaVA separately if you run `RID.py` or `IAI.py`.

## Dataset

Please obtain FHM, HarM, and MAMI, and place them in the following directories: 
```
MIND/
├── data/
│   ├── FHM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   ├── HarM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   └── MAMI/
│       ├── images/
│       │   └── ...
│       ├── test.jsonl
│       └── train.jsonl
└── ...
```

## MAMI `training.csv` analysis

Quick profile of `data/MAMI/training.csv` (tab-separated file):

- **Rows:** 10,000
- **Columns:** `file_name`, `misogynous`, `shaming`, `stereotype`, `objectification`, `violence`, `Text Transcription`

Label distribution:

- **`misogynous=1`:** 5,000
- **`misogynous=0`:** 5,000
- The primary label is perfectly balanced (50/50).

Subtype counts (multi-label dimensions):

- **`shaming`:** 1,274
- **`stereotype`:** 2,810
- **`objectification`:** 2,202
- **`violence`:** 953

Additional observations:

- **Rows with all subtype flags = 0:** 5,244  
  (includes non-misogynous rows and misogynous rows without subtype annotation)
- **Rows with more than one subtype active:** 1,863  
  (confirms non-trivial co-occurrence among subtype labels)
- **Empty `file_name`:** 0
- **Empty `Text Transcription`:** 0
- **Maximum text length:** 1,654 characters

Practical implication for this repo:

- The file is clean and complete enough for conversion to the project JSONL format.
- Because subtype columns are sparse and overlapping, they are best treated as auxiliary signals, while `misogynous` remains the main binary target used by the current pipeline.

## Quick Start

Run the three stages in order. By default, each script processes `FHM`, `HarM`, and `MAMI` via the `datasets_to_process` list at the bottom of the file. There is **no** command-line argument parsing; change those lists (and the LLaVA `model_path` in `RID.py` / `IAI.py`, currently `liuhaotian/llava-v1.5-13b`) in code if you need different settings.

1. Similar Sample Retrieval
```
python SSR.py \
--datasets HarM FHM MAMI
```

2. Relevant Insight Derivation
```
python RID.py \
--model_path liuhaotian/llava-v1.5-13b \
--datasets HarM FHM MAMI
```

3. Insight-Augmented Inference
```
python IAI.py \
--model_path liuhaotian/llava-v1.5-13b \
--datasets HarM FHM MAMI
```

Optional: use the pre-computed `SSR/*.jsonl` and `RID/*.jsonl` files shipped under `SSR/` and `RID/` to skip recomputing those stages (see the README files in those folders).
