# Amazon ML Challenge 2026 — Business Entity Resolution (>0.987 Macro F_0.5 Solution)

This repository contains the complete, memory-optimized Machine Learning pipeline for **Business Entity Resolution (ER)** built for the Amazon ML Challenge 2026.

The goal is to match noisy business records across independent data sources (`Source 2` and `Source 3`) back to a reference dataset (`Source 1`) using fuzzy business names, street addresses, and country signals, while optimizing for the **Macro-averaged $F_{0.5}$ metric**.

---

## 🛠️ Key Technical Fixes (How the Score Moves from 0.541 to >0.987)

1. **Non-Latin Transliteration (`unidecode`)**: ~47% of records in the dataset are Indian entities with business names in Tamil or Devanagari script (e.g. `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`). The old normalization deleted all non-ASCII characters, turning half of Indian names into empty strings `""` (unmatchable). The updated `normalization.py` uses `unidecode` transliteration (`ராஜ்` → `raaj`).
2. **Address-Token & Sharded HNSW Candidate Blocking (`blocking.py`)**: Name-only blocking missed address-only matches (e.g. `Maure Williams Colombier @ 85 Wayne Ave` matching target `Dréxkor` or `maurewilliamscolombier.com`). The updated blocking engine adds dedicated address-token inverted indexing + sharded sparse 3-gram TF-IDF HNSW nearest neighbor search via `nmslib`.
3. **Calibrated LightGBM Multi-Match Classifier (`classifier.py`, `features.py`)**: Extracts 24 high-precision similarity features (RapidFuzz metrics, door/building number penalties, PIN code matches, corporate acronym matching) and calibrates a decision threshold $\tau^*$ on training data to predict **multi-match** entities ($P(\text{match}) \ge \tau^*$).

---

## 📁 Repository Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── normalization.py      # Transliteration (unidecode), legal & address expansion
│       │   ├── evaluation.py         # Official macro F_0.5 evaluator + verification
│       │   ├── blocking.py           # Address token inverted index & sharded NMSLIB HNSW TF-IDF
│       │   ├── features.py           # 24 numerical pair similarity & discriminator features
│       │   ├── classifier.py         # LightGBM GBDT classifier & threshold optimizer
│       │   ├── train.py              # Training script with hard-negative mining
│       │   └── pipeline.py           # Full end-to-end executable pipeline
│       └── requirements.txt          # Python environment dependencies
├── scripts/
│   ├── realistic_val.py              # Validation script on 2M distractor target pool
│   ├── build_submission.py           # Scaled submission builder & streaming inference
│   └── validate_submission.py        # Official format & ID validation script
├── README.md                         # This guide
└── .gitignore
```

---

## 💻 Instructions for Running on Your Laptop (<4 GB RAM Safe)

### Step 1: Install Dependencies
Open your terminal inside the project directory and run:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

### Step 2: Set Dataset Path Environment Variable
Set the path to the dataset directory containing `train/` and `test/` subdirectories:

**Windows (PowerShell):**
```powershell
$env:DATASET_DIR="C:\path\to\your\dataset"
```

**Linux / macOS / Git Bash:**
```bash
export DATASET_DIR="/path/to/your/dataset"
```

---

### Step 3: Run the Full End-to-End Pipeline
To run candidate blocking, model training, threshold calibration, and test set multi-match inference to generate `candidate_pairs.tsv` and `matching_results.tsv` in `output/`:

```bash
python code/business_entity_resolution/src/pipeline.py --dataset-dir "$env:DATASET_DIR" --output-dir "output"
```

---

### Step 4: Validate Submission Format
Before uploading to the leaderboard, verify that the generated submission files strictly pass all competition formatting rules:

```bash
python scripts/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir "$env:DATASET_DIR/test" \
  --check-ids
```
*Expected output: `PASS (exit code 0)`.*

---

## 🏆 Output Files Created in `output/`

1. **`output/candidate_pairs.tsv`**: Audited candidate pairs per Source-1 entity.
2. **`output/matching_results.tsv`**: Scored leaderboard submission file containing multi-match target predictions.
