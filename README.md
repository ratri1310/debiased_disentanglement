# Mitigating Spurious Correlations for Zero-Shot Biomedical Text Classification

Implementation of the paper "Mitigating Spurious Correlations for Improved Zero-Shot Biomedical Text Classification "

## Overview

This repository implements a novel framework for zero-shot biomedical text classification that:
<img width="4372" height="2502" alt="image" src="https://github.com/user-attachments/assets/ffe1cd4d-fdaf-4158-b41b-3e9f1fe7bdbb" />

1. **Identifies spurious correlations** using UMLS knowledge graphs
2. **Debiases text representations** via translation operations
3. **Disentangles features** separating stable biomedical semantics from stochastic linguistic variance
4. **Enables zero-shot generalization** to unseen biomedical concepts

### Key Components

- **Knowledge Graph Construction**: Build label-centric knowledge graphs from UMLS to detect spurious MeSH codes
- **Translation-Based Debiasing**: Remove spurious influences while preserving semantic structure
- **Variational Disentanglement**: Separate content features (zI) from variance features (zV)
- **Contrastive Learning**: Enforce invariance to spurious correlations and linguistic nuances
- **Zero-Shot Inference**: Match test samples to unseen labels via cosine similarity


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
D000003,Abbreviations as Topic
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

### Monitor Training

Training logs show:
- Total loss (Equation 9)
- Reconstruction loss (Equation 5)
- KL divergence loss (Equation 6)
- Classification loss
- Contrastive loss (Equation 8)

```
Epoch 1/50
Train - Loss: 2.3456, Recon: 0.8234, KL: 0.2156, Cls: 0.9876, Contrast: 0.3190
Val - Loss: 2.1234, Recon: 0.7891, KL: 0.2034, Cls: 0.8901, Contrast: 0.3408
✓ Saved best model
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

### Output Files

**`metrics.json`**: Evaluation metrics
```json
{
  "f1_macro": 0.6000,
  "f1_micro": 0.5876,
  "mAP": 0.6780,
  "auc_macro": 0.9710,
  "mrr": 0.3890,
  "precision@5": 0.7234,
  "precision@10": 0.6891,
  "precision@15": 0.6543,
  "recall@5": 0.4123,
  "recall@10": 0.5234,
  "recall@15": 0.6012
}
```

**`predictions.json`**: Top-k predictions per sample
```json
[
  {
    "pmid": "34807897",
    "predicted_labels": ["D000328", "D005260", "D008297", ...],
    "scores": [0.8234, 0.7891, 0.7456, ...]
  },
  ...
]
```

## Evaluation Metrics

The framework reports standard metrics for multi-label classification:

- **F1-Score** (macro/micro): Harmonic mean of precision and recall
- **mAP** (mean Average Precision): Ranking quality across labels
- **AUC** (Area Under ROC Curve): Discrimination ability
- **MRR** (Mean Reciprocal Rank): Position of first correct label
- **Precision@K**: Precision at top-K predictions
- **Recall@K**: Recall at top-K predictions


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work is supported by the U.S. National Science Foundation (NSF) and National Institute of Health (NIH)
- PubMedBERT model from Microsoft Research
- UMLS from the National Library of Medicine
- MeSH vocabulary from NLM
