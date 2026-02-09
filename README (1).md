# Mitigating Spurious Correlations for Zero-Shot Biomedical Text Classification

Official PyTorch implementation of the paper "Mitigating Spurious Correlations for Improved Zero-Shot Biomedical Text Classification" published in Bioinformatics.

## Overview

This repository implements a novel framework for zero-shot biomedical text classification that:

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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Biomedical Abstract                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   PubMedBERT Text Encoder      │
        │   Extract [CLS] embedding z'   │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Knowledge Graph Analysis      │
        │  Identify Spurious Concepts    │
        │  Compute vspur direction       │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Translation-Based Debiasing   │
        │  z = z' - γλ·vspur             │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Variational Encoder           │
        │  → Content features (zI)       │
        │  → Variance params (μ, σ²)     │
        └────────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────┐           ┌──────────────┐
    │   zI    │           │ zV ~ N(μ,σ²) │
    └────┬────┘           └──────┬───────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Decoder: Reconstruct z        │
        │  + Classification Head         │
        │  + Contrastive Regularization  │
        └────────────────────────────────┘
```

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
    --limit 10000  # Optional: limit number of abstracts for testing
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

### Full Training Command with Hyperparameters

```bash
python train.py \
    --kg-output-dir ./kg_outputs \
    --database-json /path/to/database.json \
    --output-dir ./checkpoints \
    --pretrained-model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --hidden-dim 512 \
    --latent-dim 256 \
    --num-classes 2574 \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --gamma 1.0 \
    --beta 1.0 \
    --alpha 1.0 \
    --lambda-contrast 0.5 \
    --temperature 0.07 \
    --num-augmentations 5 \
    --device cuda \
    --num-workers 4 \
    --seed 42
```

### Hyperparameter Guide

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--gamma` | Debiasing strength | 1.0 | [0, 1] |
| `--beta` | KL divergence weight | 1.0 | [0.1, 2.0] |
| `--alpha` | Classification loss weight | 1.0 | [0.5, 2.0] |
| `--lambda-contrast` | Contrastive loss weight | 0.5 | [0.1, 1.0] |
| `--temperature` | Contrastive temperature | 0.07 | [0.05, 0.1] |
| `--num-augmentations` | Augmentations per sample | 5 | [3, 10] |

### Multi-GPU Training

```bash
# Use DataParallel for multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --kg-output-dir ./kg_outputs \
    --database-json /path/to/database.json \
    --output-dir ./checkpoints \
    --batch-size 64  # Increase batch size for multiple GPUs
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

### With FAISS for Large Label Spaces

For datasets with >10,000 unseen labels, use FAISS:

```bash
python inference.py \
    --model-checkpoint ./checkpoints/best_model.pt \
    --kg-output-dir ./kg_outputs \
    --database-json /path/to/test_database.json \
    --unseen-labels-csv ./unseen_labels.csv \
    --output-dir ./results \
    --batch-size 32 \
    --top-k 15 \
    --use-faiss \
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

## Repository Structure

```
debiased_disentanglement/
├── build_kg.py              # Knowledge graph construction
├── model.py                 # Model architecture
├── train.py                 # Training script
├── inference.py             # Zero-shot inference
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── checkpoints/            # Saved models (created during training)
    ├── best_model.pt
    └── checkpoint_epoch_*.pt
```

## GPU Memory Requirements

| Batch Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| 8 | ~12 GB | Tesla T4 |
| 16 | ~20 GB | NVIDIA A30 |
| 32 | ~38 GB | NVIDIA A100 |

**Memory optimization tips:**
- Reduce `--batch-size` if OOM errors occur
- Reduce `--num-augmentations` (trades off contrastive learning quality)
- Use gradient checkpointing (modify model.py)
- Use mixed precision training (add to train.py)

## Reproducing Paper Results

### Neurology Dataset

```bash
# 1. Build KG
python build_kg.py \
    --umls-dir /path/to/UMLS/META \
    --mesh-xlsx /path/to/meshcodes.xlsx \
    --database-json /path/to/neurology_database.json \
    --output-dir ./kg_outputs/neurology

# 2. Train
python train.py \
    --kg-output-dir ./kg_outputs/neurology \
    --database-json /path/to/neurology_database.json \
    --output-dir ./checkpoints/neurology \
    --num-classes 2574 \
    --batch-size 16 \
    --epochs 50 \
    --seed 42

# 3. Evaluate
python inference.py \
    --model-checkpoint ./checkpoints/neurology/best_model.pt \
    --kg-output-dir ./kg_outputs/neurology \
    --database-json /path/to/neurology_test.json \
    --unseen-labels-csv ./neurology_unseen_labels.csv \
    --output-dir ./results/neurology
```

Repeat for Immunology and Embryology datasets.

### Expected Results

| Dataset | F1 | mAP | AUC | MRR |
|---------|--------|---------|---------|---------|
| Neurology | 60.0 | 67.8 | 97.1 | 38.9 |
| Immunology | 59.1 | 67.0 | 97.0 | 38.4 |
| Embryology | 57.6 | 65.5 | 96.6 | 37.6 |

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size
python train.py --batch-size 8 ...

# Or reduce number of augmentations
python train.py --num-augmentations 3 ...
```

**2. FAISS not available**
```bash
# Install FAISS with GPU support
pip install faiss-gpu

# Or use CPU version
pip install faiss-cpu
```

**3. Slow training**
```bash
# Increase number of workers
python train.py --num-workers 8 ...

# Use mixed precision (requires code modification)
# Add to train.py: from torch.cuda.amp import autocast, GradScaler
```

**4. PubMedBERT download fails**
```bash
# Pre-download the model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mukherjee2024mitigating,
  title={Mitigating Spurious Correlations for Improved Zero-Shot Biomedical Text Classification},
  author={Mukherjee, Ratri and Dahal, Shailesh and Jha, Kishlay},
  journal={Bioinformatics},
  year={2024},
  publisher={Oxford University Press}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: ratri-mukherjee@uiowa.edu

## Acknowledgments

- This work is supported by the U.S. National Science Foundation (NSF) and National Institute of Health (NIH)
- PubMedBERT model from Microsoft Research
- UMLS from the National Library of Medicine
- MeSH vocabulary from NLM

## Related Work

- **BERTMeSH**: Deep contextual representation learning for MeSH indexing
- **Con-Aware**: Context-aware contrastive representation learning
- **Gen-Z**: Generative zero-shot text classification

## Future Work

- [ ] Support for other biomedical ontologies (ICD, SNOMED-CT)
- [ ] Multi-modal extensions (images + text)
- [ ] Few-shot learning capabilities
- [ ] Interpretability visualizations
- [ ] Pre-trained model checkpoints

## Version History

- **v1.0.0** (2024): Initial release
  - Knowledge graph-guided spurious detection
  - Translation-based debiasing
  - Variational disentanglement
  - Zero-shot inference with FAISS support
