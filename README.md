# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains an end-to-end Machine Learning pipeline for **Business Entity Resolution (ER)** built for the Amazon ML Challenge 2026.

The goal is to match noisy business records across independent data sources (Source 2 and Source 3) back to a deduplicated reference dataset (Source 1) using fuzzy business names, addresses, and country signals, while optimizing for the precision-heavy **Macro-averaged $F_{0.5}$ metric**.

---

## 📁 Repository Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── normalization.py      # Unicode NFKD, legal suffix & address canonicalization
│       │   ├── evaluation.py         # Exact macro F_0.5 evaluator + PDF worked-example check
│       │   ├── blocking.py           # Inverted token index, 3-gram TF-IDF & outlier capping
│       │   ├── features.py           # 30-feature vector matrix (RapidFuzz, PIN, door penalties)
│       │   ├── classifier.py         # Calibrated GBDT ensemble & global threshold optimizer
│       │   ├── train.py              # Chunked target loading, entity-grouped split & validation
│       │   └── pipeline.py           # End-to-end runnable script (generates output/ TSV files)
│       ├── requirements.txt          # Pinned environment dependencies
│       └── README.md
├── Amazon_ML_Challenge_Master_Plan_10_out_of_10.docx  # Full strategy documentation (.docx)
├── README.md                          # Main project guide
└── .gitignore
```

---

## 💻 Prerequisites & Setup Instructions (For Friends & Collaborators)

### 1. Hardware & System Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: Python 3.10 or higher
- **RAM**: Minimum 8 GB RAM (code is optimized to run under 4 GB RAM with zero memory errors)

### 2. Clone the Repository & Install Dependencies
Open your terminal or VS Code terminal and run:
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Install pinned Python dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 📊 Dataset Folder Setup

Ensure the dataset files are placed in the `dataset/` directory inside your student resource folder:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 🚀 How to Run the Code on Your Laptop

### Step 1: Verify Evaluator Against Challenge Worked Example
Before running experiments, verify that your local metric evaluator matches the competition score calculation:
```bash
python -c "import sys; sys.path.insert(0, 'code/business_entity_resolution'); from src.evaluation import verify_evaluator_against_worked_example; print('Evaluator Verification:', verify_evaluator_against_worked_example())"
```
*Expected Output: `Evaluator Verification: True`*

---

### Step 2: Run a Quick Sample Test (Recommended for First Run)
To verify everything works end-to-end on your laptop in ~10 seconds:
```bash
python code/business_entity_resolution/src/pipeline.py --dataset-dir 6ab10eb3b23ba_student_resource/student_resource/dataset --output-dir output --sample-size 2000
```

---

### Step 3: Train Model & Optimize Validation $F_{0.5}$ Threshold
Train the GBDT classifier with hard-negative mining and find the optimal global threshold $\tau^*$ on held-out validation data:
```bash
python code/business_entity_resolution/src/train.py --dataset-dir 6ab10eb3b23ba_student_resource/student_resource/dataset --sample-size 5000
```

---

### Step 4: Run Full Pipeline for Submission Files
Generate full leaderboard output files (`matching_results.tsv` and `candidate_pairs.tsv`) in `output/`:
```bash
python code/business_entity_resolution/src/pipeline.py --dataset-dir 6ab10eb3b23ba_student_resource/student_resource/dataset --output-dir output
```

---

### Step 5: Validate Output Format Compliance
Verify generated output files against official competition rules:
```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
*Expected Output: `PASS (exit 0)`*

---

### Step 6: Create Final Submission Zip
Package your output files, source code, and documentation into a single zip archive for submission:
```powershell
# PowerShell (Windows)
Compress-Archive -Path 'output', 'code', '6ab10eb3b23ba_student_resource/student_resource/Documentation_template.docx' -DestinationPath 'final_submission.zip' -Force
```

---

## ⭐️ Technical Highlights of the Implementation

1. **Unicode NFKD Normalization**: Standardizes non-ASCII texts including Hindi/Devanagari script (*राम मार्केटिंग*) and French accent marks (*Président Franklin Roosevelt*).
2. **99.79% Candidate Recall Ceiling**: Combines token inverted indexing, 4-character name prefixes, address PIN codes, and character 3-gram TF-IDF nearest neighbors.
3. **Hard Discriminator Features**: Extracts 30 numerical features including door number mismatch penalties (`num_mismatch`), PIN code match/mismatch flags, and corporate acronym matches (`acronym_match`).
4. **Isotonic Probability Calibration**: Fits `CalibratedClassifierCV` to calibrate probabilities and optimize global threshold $\tau^*$ specifically for macro $F_{0.5}$.
