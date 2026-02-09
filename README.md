# Mitigating Spurious Correlations for Zero-Shot Biomedical Text Classification

Implementation of the paper "Mitigating Spurious Correlations for Improved Zero-Shot Biomedical Text Classification "

## Overview

This repository implements a novel framework for zero-shot biomedical text classification that:
<img width="4372" height="2502" alt="image" src="https://github.com/user-attachments/assets/ffe1cd4d-fdaf-4158-b41b-3e9f1fe7bdbb" />

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 24GB+ GPU memory (recommended: NVIDIA A30 or better)

### Setup

```bash
# Clone the repository
git clone https://github.com/ratri1310/debiased_disentanglement.git
cd debiased_disentanglement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support with FAISS
pip install faiss-gpu

# For CPU-only (if no GPU available)
pip install faiss-cpu
```

## Data Preparation

### Step 1: Build Knowledge Graph

First, construct the label-centric knowledge graph to identify spurious correlations:

```bash
python build_kg.py \
    --umls-dir /path/to/UMLS/META \
    --mesh-xlsx /path/to/meshcodes.xlsx \
    --database-json /path/to/neurology_database.json \
    --output-dir ./kg_outputs \
```

**Required Files:**
- `UMLS/META/`: UMLS Metathesaurus files (MRCONSO.RRF, MRSTY.RRF, MRREL.RRF)
- `meshcodes.xlsx`: MeSH vocabulary with columns: `Unique ID`, `MeSH Heading`, `Tree Number(s)`
- `database.json`: PubMed articles with MeSH annotations

**Outputs:**
- `nodes.csv`: Graph nodes with MeSH/CUI mappings
- `edges.csv`: UMLS relationships between concepts
- `spurious_detection.json`: Spurious vs. connected codes per abstract
- `maps.json`: ID mappings for training
- `relationship_types.json`: Statistics on relationship types

### Step 2: Prepare Train/Val/Test Splits

Split your data into training, validation, and test sets. The test set should contain unseen labels for zero-shot evaluation.

### Step 3: Create Unseen Labels CSV

Create a CSV file with unseen label information:

```csv
label_id,descriptor_text
D000001,Calcimycin
D000002,Temefos
D000003, Abbreviations as Topic
...
```

## Training

### Basic Training

```bash
python train.py \
    --kg-output-dir ./kg_outputs \
    --database-json /path/to/database.json \
    --output-dir ./checkpoints \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --device cuda
```

## Zero-Shot Inference

### Basic Inference

```bash
python inference.py \
    --model-checkpoint ./checkpoints/best_model.pt \
    --kg-output-dir ./kg_outputs \
    --database-json /path/to/test_database.json \
    --unseen-labels-csv ./unseen_labels.csv \
    --output-dir ./results \
    --batch-size 32 \
    --top-k 15 \
    --device cuda
```

## Evaluation Metrics

The framework reports standard metrics for multi-label classification:

- **F1-Score** (macro/micro): Harmonic mean of precision and recall
- **mAP** (mean Average Precision): Ranking quality across labels
- **AUC** (Area Under ROC Curve): Discrimination ability
- **MRR** (Mean Reciprocal Rank): Position of first correct label


## License

This project is licensed under the MIT License - see the LICENSE file for details.
