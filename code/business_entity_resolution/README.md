# Business Entity Resolution Package

This package contains the core Python source code and execution scripts for the Amazon ML Challenge 2026.

## Quick Start Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Verify Evaluator
```bash
python -c "import sys; sys.path.insert(0, '.'); from src.evaluation import verify_evaluator_against_worked_example; print(verify_evaluator_against_worked_example())"
```

### 3. Run Pipeline
```bash
python src/pipeline.py --dataset-dir <path_to_dataset> --output-dir ../../output
```

### 4. Train Model
```bash
python src/train.py --dataset-dir <path_to_dataset> --sample-size 5000
```
