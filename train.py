"""
Training script for Debiased Disentanglement Model

Implements the unified training objective (Equation 9 in paper):
Ltotal = LVAE + α·Lcls + λ·Lcontrast
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Optional
import pandas as pd

from model import (
    DebiasedDisentanglementModel,
    ContrastiveLoss,
    compute_reconstruction_loss,
    compute_kl_divergence
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BiomedicalDataset(Dataset):
    """
    Dataset for biomedical text classification with spurious correlation information.
    """
    
    def __init__(
        self,
        abstracts: List[str],
        labels: List[List[int]],
        spurious_concepts: Optional[List[List[str]]] = None,
        num_classes: int = 2574
    ):
        """
        Args:
            abstracts: List of abstract texts
            labels: List of label indices for each abstract
            spurious_concepts: List of spurious concept descriptor names for each abstract
            num_classes: Total number of classes
        """
        self.abstracts = abstracts
        self.labels = labels
        self.spurious_concepts = spurious_concepts
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.abstracts)
    
    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        
        # Convert label indices to multi-hot vector
        label_vec = torch.zeros(self.num_classes)
        if self.labels[idx]:
            label_vec[self.labels[idx]] = 1.0
        
        # Get spurious concepts if available
        spurious = self.spurious_concepts[idx] if self.spurious_concepts else []
        
        # Return class index for contrastive loss (use first label as proxy)
        class_idx = self.labels[idx][0] if self.labels[idx] else 0
        
        return {
            'abstract': abstract,
            'label_vec': label_vec,
            'class_idx': class_idx,
            'spurious': spurious
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length data."""
    abstracts = [item['abstract'] for item in batch]
    label_vecs = torch.stack([item['label_vec'] for item in batch])
    class_indices = torch.tensor([item['class_idx'] for item in batch])
    spurious_concepts = [item['spurious'] for item in batch]
    
    return {
        'abstracts': abstracts,
        'label_vecs': label_vecs,
        'class_indices': class_indices,
        'spurious_concepts': spurious_concepts
    }


def load_data_from_kg_outputs(
    kg_output_dir: str,
    database_json: str,
    split: str = 'train'
) -> tuple:
    """
    Load training data from knowledge graph outputs and database.
    
    Args:
        kg_output_dir: Directory containing KG output files
        database_json: Path to the database JSON file
        split: 'train', 'val', or 'test'
        
    Returns:
        abstracts, labels, spurious_concepts, mesh_to_id mapping
    """
    kg_dir = Path(kg_output_dir)
    
    # Load spurious detection results
    with open(kg_dir / 'spurious_detection.json', 'r') as f:
        spurious_data = json.load(f)
    
    # Load maps
    with open(kg_dir / 'maps.json', 'r') as f:
        maps = json.load(f)
        mesh_to_cui = maps['mesh_to_cui']
        cui_to_id = maps['cui_to_id']
    
    # Create mesh_to_id mapping
    mesh_to_id = {}
    for mesh, cui in mesh_to_cui.items():
        if cui in cui_to_id:
            mesh_to_id[mesh] = cui_to_id[cui]
    
    # Load database
    with open(database_json, 'r') as f:
        raw_data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(raw_data, list):
        data = raw_data
    elif isinstance(raw_data, dict):
        if 'articles' in raw_data:
            data = raw_data['articles']
        elif 'data' in raw_data:
            data = raw_data['data']
        else:
            data = list(raw_data.values())
    
    # Load nodes to get concept names
    nodes_df = pd.read_csv(kg_dir / 'nodes.csv')
    mesh_to_name = dict(zip(nodes_df['mesh_id'], nodes_df['pref_name']))
    
    # Process data
    abstracts = []
    labels = []
    spurious_concepts = []
    
    for article in tqdm(data, desc=f"Loading {split} data"):
        pmid = str(article['pmid'])
        
        # Skip if not in spurious detection results
        if pmid not in spurious_data:
            continue
        
        # Get abstract text
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
        
        # Convert to internal IDs
        label_ids = [mesh_to_id[m] for m in mesh_codes if m in mesh_to_id]
        
        if not label_ids:
            continue
        
        # Get spurious concept names
        spurious_mesh = spurious_data[pmid]['spurious_codes']
        spurious_names = [mesh_to_name.get(m, m) for m in spurious_mesh if m in mesh_to_name]
        
        abstracts.append(full_text)
        labels.append(label_ids)
        spurious_concepts.append(spurious_names)
    
    logger.info(f"Loaded {len(abstracts)} {split} samples")
    
    return abstracts, labels, spurious_concepts, mesh_to_id


def train_epoch(
    model: DebiasedDisentanglementModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    contrastive_loss_fn: ContrastiveLoss,
    alpha: float,
    lambda_contrast: float,
    device: str,
    num_augmentations: int = 5
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        contrastive_loss_fn: Contrastive loss function
        alpha: Weight for classification loss
        lambda_contrast: Weight for contrastive loss
        device: Device to run on
        num_augmentations: Number of augmentations for contrastive learning
        
    Returns:
        Dictionary of average losses
    """
    model.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_cls_loss = 0
    total_contrast_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        abstracts = batch['abstracts']
        label_vecs = batch['label_vecs'].to(device)
        class_indices = batch['class_indices'].to(device)
        spurious_concepts = batch['spurious_concepts']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            texts=abstracts,
            spurious_concepts=spurious_concepts,
            labels=label_vecs,
            return_augmentations=True,
            num_augmentations=num_augmentations
        )
        
        # Reconstruction loss (Equation 5)
        recon_loss = compute_reconstruction_loss(
            outputs['z_debiased'],
            outputs['z_reconstructed']
        )
        
        # KL divergence (Equation 6)
        kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
        
        # VAE loss (Equation 6)
        vae_loss = recon_loss + model.beta * kl_loss
        
        # Classification loss
        cls_loss = nn.BCEWithLogitsLoss()(outputs['logits'], label_vecs)
        
        # Contrastive loss (Equation 8)
        contrast_loss = contrastive_loss_fn(
            outputs['augmentations'],
            class_indices
        )
        
        # Total loss (Equation 9)
        loss = vae_loss + alpha * cls_loss + lambda_contrast * contrast_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_cls_loss += cls_loss.item()
        total_contrast_loss += contrast_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'contrast_loss': total_contrast_loss / num_batches
    }


def validate(
    model: DebiasedDisentanglementModel,
    dataloader: DataLoader,
    contrastive_loss_fn: ContrastiveLoss,
    alpha: float,
    lambda_contrast: float,
    device: str
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_cls_loss = 0
    total_contrast_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            abstracts = batch['abstracts']
            label_vecs = batch['label_vecs'].to(device)
            class_indices = batch['class_indices'].to(device)
            spurious_concepts = batch['spurious_concepts']
            
            # Forward pass
            outputs = model(
                texts=abstracts,
                spurious_concepts=spurious_concepts,
                labels=label_vecs,
                return_augmentations=True,
                num_augmentations=5
            )
            
            # Compute losses
            recon_loss = compute_reconstruction_loss(
                outputs['z_debiased'],
                outputs['z_reconstructed']
            )
            kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
            vae_loss = recon_loss + model.beta * kl_loss
            cls_loss = nn.BCEWithLogitsLoss()(outputs['logits'], label_vecs)
            contrast_loss = contrastive_loss_fn(
                outputs['augmentations'],
                class_indices
            )
            loss = vae_loss + alpha * cls_loss + lambda_contrast * contrast_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_cls_loss += cls_loss.item()
            total_contrast_loss += contrast_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'contrast_loss': total_contrast_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train Debiased Disentanglement Model'
    )
    
    # Data arguments
    parser.add_argument('--kg-output-dir', type=str, required=True,
                       help='Directory containing knowledge graph outputs')
    parser.add_argument('--database-json', type=str, required=True,
                       help='Path to database JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--pretrained-model', type=str,
                       default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                       help='Pretrained language model')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=256,
                       help='Latent dimension')
    parser.add_argument('--num-classes', type=int, default=2574,
                       help='Number of training classes')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Debiasing strength')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL divergence weight')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Classification loss weight')
    parser.add_argument('--lambda-contrast', type=float, default=0.5,
                       help='Contrastive loss weight')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    parser.add_argument('--num-augmentations', type=int, default=5,
                       help='Number of augmentations (K)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading training data...")
    train_abstracts, train_labels, train_spurious, mesh_to_id = load_data_from_kg_outputs(
        args.kg_output_dir,
        args.database_json,
        split='train'
    )
    
    logger.info("Loading validation data...")
    val_abstracts, val_labels, val_spurious, _ = load_data_from_kg_outputs(
        args.kg_output_dir,
        args.database_json,
        split='val'
    )
    
    # Update num_classes based on actual data
    actual_num_classes = max(max(labels) for labels in train_labels + val_labels) + 1
    args.num_classes = actual_num_classes
    logger.info(f"Number of classes: {args.num_classes}")
    
    # Create datasets
    train_dataset = BiomedicalDataset(
        train_abstracts,
        train_labels,
        train_spurious,
        args.num_classes
    )
    
    val_dataset = BiomedicalDataset(
        val_abstracts,
        val_labels,
        val_spurious,
        args.num_classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = DebiasedDisentanglementModel(
        pretrained_model=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        gamma=args.gamma,
        beta=args.beta,
        temperature=args.temperature,
        device=device
    )
    
    # Initialize contrastive loss
    contrastive_loss_fn = ContrastiveLoss(temperature=args.temperature)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            contrastive_loss_fn,
            args.alpha,
            args.lambda_contrast,
            device,
            args.num_augmentations
        )
        
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Recon: {train_metrics['recon_loss']:.4f}, "
            f"KL: {train_metrics['kl_loss']:.4f}, "
            f"Cls: {train_metrics['cls_loss']:.4f}, "
            f"Contrast: {train_metrics['contrast_loss']:.4f}"
        )
        
        # Validate
        val_metrics = validate(
            model,
            val_loader,
            contrastive_loss_fn,
            args.alpha,
            args.lambda_contrast,
            device
        )
        
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Recon: {val_metrics['recon_loss']:.4f}, "
            f"KL: {val_metrics['kl_loss']:.4f}, "
            f"Cls: {val_metrics['cls_loss']:.4f}, "
            f"Contrast: {val_metrics['contrast_loss']:.4f}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            logger.info("✓ Saved best model")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        scheduler.step()
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
