#!/bin/bash

# Quick Start Script for Debiased Disentanglement Framework
# This script runs the complete pipeline from KG construction to zero-shot inference

set -e  # Exit on error

echo "======================================================================"
echo "Debiased Disentanglement Framework - Quick Start"
echo "======================================================================"

# Configuration
UMLS_DIR=${UMLS_DIR:-"/path/to/UMLS/META"}
MESH_XLSX=${MESH_XLSX:-"/path/to/meshcodes.xlsx"}
DATABASE_JSON=${DATABASE_JSON:-"/path/to/database.json"}
OUTPUT_BASE=${OUTPUT_BASE:-"./outputs"}
DEVICE=${DEVICE:-"cuda"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-50}

# Create output directories
KG_OUTPUT_DIR="${OUTPUT_BASE}/kg_outputs"
CHECKPOINT_DIR="${OUTPUT_BASE}/checkpoints"
DATA_SPLITS_DIR="${OUTPUT_BASE}/data_splits"
RESULTS_DIR="${OUTPUT_BASE}/results"

mkdir -p ${KG_OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATA_SPLITS_DIR}
mkdir -p ${RESULTS_DIR}

echo ""
echo "Configuration:"
echo "  UMLS Directory: ${UMLS_DIR}"
echo "  MeSH Excel: ${MESH_XLSX}"
echo "  Database JSON: ${DATABASE_JSON}"
echo "  Output Directory: ${OUTPUT_BASE}"
echo "  Device: ${DEVICE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo ""

# Step 1: Build Knowledge Graph
echo "======================================================================"
echo "Step 1: Building Knowledge Graph"
echo "======================================================================"
python build_kg.py \
    --umls-dir ${UMLS_DIR} \
    --mesh-xlsx ${MESH_XLSX} \
    --database-json ${DATABASE_JSON} \
    --output-dir ${KG_OUTPUT_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Knowledge graph construction failed!"
    exit 1
fi

echo "✓ Knowledge graph built successfully"
echo ""

# Step 2: Create train/val/test splits
echo "======================================================================"
echo "Step 2: Creating Train/Val/Test Splits"
echo "======================================================================"
python utils.py \
    --command split \
    --database-json ${DATABASE_JSON} \
    --output-dir ${DATA_SPLITS_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Data splitting failed!"
    exit 1
fi

echo "✓ Data splits created successfully"
echo ""

# Step 3: Extract unseen labels
echo "======================================================================"
echo "Step 3: Extracting Unseen Labels"
echo "======================================================================"
python utils.py \
    --command unseen \
    --kg-output-dir ${KG_OUTPUT_DIR} \
    --output-dir ${DATA_SPLITS_DIR}

if [ $? -ne 0 ]; then
    echo "Error: Unseen label extraction failed!"
    exit 1
fi

echo "✓ Unseen labels extracted successfully"
echo ""

# Step 4: Analyze spurious correlations
echo "======================================================================"
echo "Step 4: Analyzing Spurious Correlations"
echo "======================================================================"
python utils.py \
    --command stats \
    --kg-output-dir ${KG_OUTPUT_DIR}

echo ""

# Step 5: Verify data integrity
echo "======================================================================"
echo "Step 5: Verifying Data Integrity"
echo "======================================================================"
python utils.py \
    --command verify \
    --database-json ${DATABASE_JSON} \
    --kg-output-dir ${KG_OUTPUT_DIR}

echo ""

# Step 6: Train model
echo "======================================================================"
echo "Step 6: Training Model"
echo "======================================================================"
python train.py \
    --kg-output-dir ${KG_OUTPUT_DIR} \
    --database-json ${DATABASE_JSON} \
    --output-dir ${CHECKPOINT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --device ${DEVICE}

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "✓ Model trained successfully"
echo ""

# Step 7: Zero-shot inference
echo "======================================================================"
echo "Step 7: Zero-Shot Inference"
echo "======================================================================"
python inference.py \
    --model-checkpoint ${CHECKPOINT_DIR}/best_model.pt \
    --kg-output-dir ${KG_OUTPUT_DIR} \
    --database-json ${DATABASE_JSON} \
    --unseen-labels-csv ${DATA_SPLITS_DIR}/unseen_labels.csv \
    --output-dir ${RESULTS_DIR} \
    --batch-size 32 \
    --top-k 15 \
    --use-faiss \
    --device ${DEVICE}

if [ $? -ne 0 ]; then
    echo "Error: Inference failed!"
    exit 1
fi

echo "✓ Inference completed successfully"
echo ""

# Step 8: Display results
echo "======================================================================"
echo "Pipeline Completed Successfully!"
echo "======================================================================"
echo ""
echo "Output locations:"
echo "  Knowledge Graph: ${KG_OUTPUT_DIR}"
echo "  Model Checkpoints: ${CHECKPOINT_DIR}"
echo "  Data Splits: ${DATA_SPLITS_DIR}"
echo "  Results: ${RESULTS_DIR}"
echo ""
echo "Key files:"
echo "  - KG nodes: ${KG_OUTPUT_DIR}/nodes.csv"
echo "  - KG edges: ${KG_OUTPUT_DIR}/edges.csv"
echo "  - Spurious detection: ${KG_OUTPUT_DIR}/spurious_detection.json"
echo "  - Best model: ${CHECKPOINT_DIR}/best_model.pt"
echo "  - Metrics: ${RESULTS_DIR}/metrics.json"
echo "  - Predictions: ${RESULTS_DIR}/predictions.json"
echo ""

# Display metrics if available
if [ -f "${RESULTS_DIR}/metrics.json" ]; then
    echo "======================================================================"
    echo "Evaluation Metrics:"
    echo "======================================================================"
    python -c "import json; print(json.dumps(json.load(open('${RESULTS_DIR}/metrics.json')), indent=2))"
    echo ""
fi

echo "======================================================================"
echo "All done! 🎉"
echo "======================================================================"
