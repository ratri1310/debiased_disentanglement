"""
Zero-Shot Inference Script

Implements Section 3.7 from the paper: zero-shot classification via
cosine similarity between content features and label prototypes.
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    label_ranking_average_precision_score
)
import faiss

from model import DebiasedDisentanglementModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_unseen_labels(label_file: str) -> Tuple[List[str], List[str]]:
    """
    Load unseen label IDs and their descriptor texts.
    
    Args:
        label_file: Path to CSV file with columns: label_id, descriptor_text
        
    Returns:
        label_ids: List of label IDs
        label_texts: List of label descriptor texts
    """
    df = pd.read_csv(label_file)
    label_ids = df['label_id'].tolist()
    label_texts = df['descriptor_text'].tolist()
    
    logger.info(f"Loaded {len(label_ids)} unseen labels")
    return label_ids, label_texts


def load_test_data(
    database_json: str,
    kg_output_dir: str,
    test_pmids: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Load test abstracts and their labels.
    
    Args:
        database_json: Path to database JSON
        kg_output_dir: Directory with KG outputs
        test_pmids: Optional list of PMIDs to use for testing
        
    Returns:
        pmids, abstracts, mesh_labels, spurious_concepts
    """
    kg_dir = Path(kg_output_dir)
    
    # Load spurious detection results
    with open(kg_dir / 'spurious_detection.json', 'r') as f:
        spurious_data = json.load(f)
    
    # Load nodes to get concept names
    nodes_df = pd.read_csv(kg_dir / 'nodes.csv')
    mesh_to_name = dict(zip(nodes_df['mesh_id'], nodes_df['pref_name']))
    
    # Load database
    with open(database_json, 'r') as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, list):
        data = raw_data
    elif isinstance(raw_data, dict):
        if 'articles' in raw_data:
            data = raw_data['articles']
        elif 'data' in raw_data:
            data = raw_data['data']
        else:
            data = list(raw_data.values())
    
    pmids = []
    abstracts = []
    mesh_labels = []
    spurious_concepts = []
    
    for article in tqdm(data, desc="Loading test data"):
        pmid = str(article['pmid'])
        
        # Filter by test PMIDs if provided
        if test_pmids and pmid not in test_pmids:
            continue
        
        # Skip if not in spurious detection
        if pmid not in spurious_data:
            continue
        
        # Get abstract
        title = article.get('title', '')
        abstract_text = article.get('abstract', '')
        full_text = f"{title}. {abstract_text}".strip()
        
        if not full_text:
            continue
        
        # Get MeSH codes
        mesh_codes = []
        if 'meshMajorEnhanced' in article:
            for entry in article['meshMajorEnhanced']:
                for key, value in entry.items():
                    if key.startswith('unique_id_') and value:
                        mesh_codes.append(value)
        
        if not mesh_codes:
            continue
        
        # Get spurious concept names
        spurious_mesh = spurious_data[pmid]['spurious_codes']
        spurious_names = [mesh_to_name.get(m, m) for m in spurious_mesh if m in mesh_to_name]
        
        pmids.append(pmid)
        abstracts.append(full_text)
        mesh_labels.append(mesh_codes)
        spurious_concepts.append(spurious_names)
    
    logger.info(f"Loaded {len(abstracts)} test samples")
    
    return pmids, abstracts, mesh_labels, spurious_concepts


def encode_test_samples(
    model: DebiasedDisentanglementModel,
    abstracts: List[str],
    spurious_concepts: List[List[str]],
    batch_size: int = 16,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Encode test abstracts to content features (zI).
    
    Args:
        model: Trained model
        abstracts: List of abstract texts
        spurious_concepts: List of spurious concept lists
        batch_size: Batch size for encoding
        device: Device to use
        
    Returns:
        content_features: Content embeddings [num_samples, dc]
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(abstracts), batch_size), desc="Encoding test samples"):
            batch_abstracts = abstracts[i:i+batch_size]
            batch_spurious = spurious_concepts[i:i+batch_size]
            
            # Forward pass
            outputs = model(
                texts=batch_abstracts,
                spurious_concepts=batch_spurious,
                return_augmentations=False
            )
            
            # Get content features
            zI = outputs['zI']  # [batch, dc]
            all_features.append(zI.cpu())
    
    content_features = torch.cat(all_features, dim=0)
    logger.info(f"Encoded {len(content_features)} test samples to content features")
    
    return content_features


def build_faiss_index(
    label_prototypes: torch.Tensor,
    use_gpu: bool = True
) -> faiss.Index:
    """
    Build FAISS index for efficient similarity search.
    
    Args:
        label_prototypes: Label prototype embeddings [num_labels, dc]
        use_gpu: Whether to use GPU for FAISS
        
    Returns:
        index: FAISS index
    """
    num_labels, dim = label_prototypes.shape
    
    # Convert to numpy
    prototypes_np = label_prototypes.cpu().numpy().astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(prototypes_np)
    
    # Create index
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
    
    # Move to GPU if requested and available
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add vectors
    index.add(prototypes_np)
    
    logger.info(f"Built FAISS index with {num_labels} label prototypes")
    
    return index


def predict_zero_shot(
    content_features: torch.Tensor,
    label_prototypes: torch.Tensor,
    k: int = 10,
    use_faiss: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform zero-shot prediction via cosine similarity.
    Equation 10 in paper.
    
    Args:
        content_features: Test content features [num_samples, dc]
        label_prototypes: Label prototypes [num_labels, dc]
        k: Number of top labels to retrieve
        use_faiss: Whether to use FAISS for efficiency
        
    Returns:
        top_k_indices: Top-k label indices [num_samples, k]
        similarities: Similarity scores [num_samples, num_labels]
    """
    num_samples = content_features.shape[0]
    num_labels = label_prototypes.shape[0]
    
    # Normalize features
    content_features_norm = F.normalize(content_features, p=2, dim=1)
    label_prototypes_norm = F.normalize(label_prototypes, p=2, dim=1)
    
    if use_faiss and num_labels > 1000:
        # Use FAISS for large label spaces
        index = build_faiss_index(label_prototypes_norm, use_gpu=torch.cuda.is_available())
        
        # Search
        features_np = content_features_norm.cpu().numpy().astype('float32')
        faiss.normalize_L2(features_np)
        
        similarities_k, top_k_indices = index.search(features_np, k)
        
        # Get full similarity matrix (for metrics)
        similarities = torch.matmul(
            content_features_norm.cpu(),
            label_prototypes_norm.cpu().T
        ).numpy()
        
    else:
        # Compute cosine similarity: [num_samples, num_labels]
        similarities = torch.matmul(
            content_features_norm.cpu(),
            label_prototypes_norm.cpu().T
        ).numpy()
        
        # Get top-k
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    
    logger.info(f"Performed zero-shot prediction for {num_samples} samples")
    
    return top_k_indices, similarities


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: List[List[str]],
    label_ids: List[str],
    k_values: List[int] = [5, 10, 15]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Similarity scores [num_samples, num_labels]
        ground_truth: List of true MeSH codes for each sample
        label_ids: List of label IDs corresponding to predictions
        k_values: k values for top-k metrics
        
    Returns:
        metrics: Dictionary of metric values
    """
    num_samples = len(ground_truth)
    num_labels = len(label_ids)
    
    # Create label_id to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_ids)}
    
    # Convert ground truth to binary matrix
    y_true = np.zeros((num_samples, num_labels))
    for i, true_labels in enumerate(ground_truth):
        for label in true_labels:
            if label in label_to_idx:
                y_true[i, label_to_idx[label]] = 1
    
    # Filter samples with at least one valid label
    valid_samples = y_true.sum(axis=1) > 0
    y_true = y_true[valid_samples]
    y_pred = predictions[valid_samples]
    
    if len(y_true) == 0:
        logger.warning("No valid samples for evaluation")
        return {}
    
    # Threshold predictions for F1
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Compute metrics
    metrics = {}
    
    # F1 Score (macro and micro)
    metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    # Mean Average Precision (mAP)
    metrics['mAP'] = label_ranking_average_precision_score(y_true, y_pred)
    
    # AUC (macro)
    try:
        metrics['auc_macro'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        metrics['auc_macro'] = 0.0
    
    # Mean Reciprocal Rank (MRR)
    mrr_sum = 0
    for i in range(len(y_true)):
        # Get indices sorted by prediction score
        sorted_indices = np.argsort(-y_pred[i])
        
        # Find rank of first true label
        for rank, idx in enumerate(sorted_indices, start=1):
            if y_true[i, idx] == 1:
                mrr_sum += 1.0 / rank
                break
    
    metrics['mrr'] = mrr_sum / len(y_true)
    
    # Precision@K
    for k in k_values:
        precisions = []
        for i in range(len(y_true)):
            top_k = np.argsort(-y_pred[i])[:k]
            num_correct = y_true[i, top_k].sum()
            precisions.append(num_correct / k)
        metrics[f'precision@{k}'] = np.mean(precisions)
    
    # Recall@K
    for k in k_values:
        recalls = []
        for i in range(len(y_true)):
            num_true = y_true[i].sum()
            if num_true == 0:
                continue
            top_k = np.argsort(-y_pred[i])[:k]
            num_correct = y_true[i, top_k].sum()
            recalls.append(num_correct / num_true)
        metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
    
    return metrics


def save_predictions(
    output_file: str,
    pmids: List[str],
    predictions: np.ndarray,
    label_ids: List[str],
    k: int = 15
):
    """
    Save top-k predictions to file.
    
    Args:
        output_file: Output file path
        pmids: List of PMIDs
        predictions: Similarity scores [num_samples, num_labels]
        label_ids: List of label IDs
        k: Number of top predictions to save
    """
    results = []
    
    for i, pmid in enumerate(pmids):
        # Get top-k predictions
        top_k_indices = np.argsort(-predictions[i])[:k]
        top_k_scores = predictions[i, top_k_indices]
        top_k_labels = [label_ids[idx] for idx in top_k_indices]
        
        results.append({
            'pmid': pmid,
            'predicted_labels': top_k_labels,
            'scores': top_k_scores.tolist()
        })
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved predictions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot inference for biomedical text classification'
    )
    
    # Model arguments
    parser.add_argument('--model-checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--kg-output-dir', type=str, required=True,
                       help='Directory containing KG outputs')
    parser.add_argument('--database-json', type=str, required=True,
                       help='Path to test database JSON')
    parser.add_argument('--unseen-labels-csv', type=str, required=True,
                       help='CSV file with unseen label IDs and descriptors')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save predictions')
    
    # Inference arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--top-k', type=int, default=15,
                       help='Number of top labels to predict')
    parser.add_argument('--use-faiss', action='store_true',
                       help='Use FAISS for efficient similarity search')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model checkpoint...")
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model_args = checkpoint['args']
    
    model = DebiasedDisentanglementModel(
        pretrained_model=model_args['pretrained_model'],
        hidden_dim=model_args['hidden_dim'],
        latent_dim=model_args['latent_dim'],
        num_classes=model_args['num_classes'],
        gamma=model_args['gamma'],
        beta=model_args['beta'],
        temperature=model_args['temperature'],
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load unseen labels
    logger.info("Loading unseen labels...")
    label_ids, label_texts = load_unseen_labels(args.unseen_labels_csv)
    
    # Encode label prototypes
    logger.info("Encoding label prototypes...")
    label_prototypes = model.encode_labels(label_texts)  # [num_labels, H]
    
    # Project to content space (using mean pooling for prototype)
    with torch.no_grad():
        # We need to get zI for labels - use encoder
        h = model.vae_encoder.encoder(label_prototypes)
        label_prototypes_zI = model.vae_encoder.content_pool(h.unsqueeze(1)).squeeze(1)
    
    # Load test data
    logger.info("Loading test data...")
    pmids, abstracts, mesh_labels, spurious_concepts = load_test_data(
        args.database_json,
        args.kg_output_dir
    )
    
    # Encode test samples
    logger.info("Encoding test samples...")
    content_features = encode_test_samples(
        model,
        abstracts,
        spurious_concepts,
        args.batch_size,
        args.device
    )
    
    # Perform zero-shot prediction
    logger.info("Performing zero-shot prediction...")
    top_k_indices, similarities = predict_zero_shot(
        content_features,
        label_prototypes_zI,
        k=args.top_k,
        use_faiss=args.use_faiss
    )
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(
        similarities,
        mesh_labels,
        label_ids,
        k_values=[5, 10, 15]
    )
    
    # Print metrics
    logger.info("\n" + "="*60)
    logger.info("ZERO-SHOT EVALUATION METRICS")
    logger.info("="*60)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    logger.info("="*60)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    logger.info("Saving predictions...")
    save_predictions(
        output_dir / 'predictions.json',
        pmids,
        similarities,
        label_ids,
        k=args.top_k
    )
    
    logger.info("\nInference completed!")


if __name__ == '__main__':
    main()
