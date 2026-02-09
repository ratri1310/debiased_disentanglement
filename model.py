"""
Mitigating Spurious Correlations for Zero-Shot Biomedical Text Classification
Main Model Architecture

This module implements the complete framework for debiasing and disentanglement
of biomedical text representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import numpy as np


class SpuriousDirectionEstimator(nn.Module):
    """
    Estimates spurious direction from isolated concept codes.
    Section 3.2 in paper.
    """
    
    def __init__(self, text_encoder: nn.Module, device: str = 'cuda'):
        super().__init__()
        self.text_encoder = text_encoder
        self.device = device
        
    def forward(self, spurious_concepts: List[str], tokenizer) -> torch.Tensor:
        """
        Compute normalized centroid of spurious concept embeddings.
        
        Args:
            spurious_concepts: List of spurious concept descriptor names
            tokenizer: Tokenizer for encoding text
            
        Returns:
            vspur: Normalized spurious direction vector [H]
        """
        if len(spurious_concepts) == 0:
            return None
            
        # Encode all spurious concepts
        embeddings = []
        for concept in spurious_concepts:
            # Tokenize concept descriptor
            inputs = tokenizer(
                concept,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get [CLS] token embedding from encoder
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, H]
                embeddings.append(cls_embedding)
        
        # Stack and compute centroid
        embeddings = torch.cat(embeddings, dim=0)  # [|Si|, H]
        centroid = embeddings.mean(dim=0)  # [H]
        
        # Normalize to unit length
        vspur = F.normalize(centroid, p=2, dim=0)  # [H]
        
        return vspur


class TranslationDebiaser(nn.Module):
    """
    Removes spurious bias via translation operation.
    Section 3.3 in paper: zi = z'i - γλi*vspur,i
    """
    
    def __init__(self, gamma: float = 1.0):
        """
        Args:
            gamma: Debiasing strength parameter, controls correction magnitude
        """
        super().__init__()
        self.gamma = gamma
        
    def forward(self, z_prime: torch.Tensor, vspur: torch.Tensor) -> torch.Tensor:
        """
        Apply translation-based debiasing.
        
        Args:
            z_prime: Original text embedding [batch, H]
            vspur: Spurious direction vector [H]
            
        Returns:
            z: Debiased embedding [batch, H]
        """
        if vspur is None:
            return z_prime
            
        # Compute projection magnitude: λi = <z'i, vspur,i>
        lambda_i = torch.sum(z_prime * vspur.unsqueeze(0), dim=1, keepdim=True)  # [batch, 1]
        
        # Translation: zi = z'i - γλi*vspur,i
        z = z_prime - self.gamma * lambda_i * vspur.unsqueeze(0)  # [batch, H]
        
        return z


class VariationalEncoder(nn.Module):
    """
    Encodes debiased representations and parameterizes variational distribution.
    Section 3.4 in paper.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, latent_dim: int = 256):
        """
        Args:
            input_dim: Input embedding dimension (H)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent dimension for content and variance features (dc)
        """
        super().__init__()
        
        # Shared encoder: 2-layer MLP with ReLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Content feature extraction (deterministic)
        self.content_pool = nn.AdaptiveAvgPool1d(latent_dim)
        
        # Variance feature parameterization (stochastic)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode debiased embedding into content and variance features.
        
        Args:
            z: Debiased embedding [batch, H]
            
        Returns:
            zI: Deterministic content features [batch, dc]
            mu: Mean of variance distribution [batch, dc]
            logvar: Log-variance of variance distribution [batch, dc]
        """
        # Shared encoding
        h = self.encoder(z)  # [batch, hidden_dim]
        
        # Content features (deterministic)
        zI = self.content_pool(h.unsqueeze(1)).squeeze(1)  # [batch, dc]
        
        # Variance distribution parameters
        mu = self.mu_layer(h)  # [batch, dc]
        logvar = self.logvar_layer(h)  # [batch, dc]
        
        return zI, mu, logvar


class VariationalDecoder(nn.Module):
    """
    Reconstructs debiased embeddings from content and variance features.
    Section 3.4 in paper.
    """
    
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, output_dim: int = 768):
        """
        Args:
            latent_dim: Latent dimension (dc)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (H)
        """
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z_combined: torch.Tensor) -> torch.Tensor:
        """
        Decode combined features back to embedding space.
        
        Args:
            z_combined: zI + zV [batch, dc]
            
        Returns:
            z_reconstructed: Reconstructed embedding [batch, H]
        """
        return self.decoder(z_combined)


class DebiasedDisentanglementModel(nn.Module):
    """
    Complete framework integrating all components.
    Implements the full pipeline from Figure 2 in paper.
    """
    
    def __init__(
        self,
        pretrained_model: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_classes: int = 2574,  # Default: Neurology dataset
        gamma: float = 1.0,
        beta: float = 1.0,
        temperature: float = 0.07,
        device: str = 'cuda'
    ):
        """
        Args:
            pretrained_model: Name or path of pretrained language model
            hidden_dim: Hidden dimension for encoder/decoder
            latent_dim: Latent dimension for disentangled features
            num_classes: Number of training classes
            gamma: Debiasing strength
            beta: KL divergence weight
            temperature: Temperature for contrastive loss
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.latent_dim = latent_dim
        
        # Load pretrained text encoder (PubMedBERT)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.text_encoder = AutoModel.from_pretrained(pretrained_model).to(device)
        
        # Freeze text encoder (as per paper - pretrained model used for embeddings)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.embedding_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
        
        # Initialize components
        self.spurious_estimator = SpuriousDirectionEstimator(
            self.text_encoder, device
        )
        
        self.debiaser = TranslationDebiaser(gamma=gamma)
        
        self.vae_encoder = VariationalEncoder(
            input_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        ).to(device)
        
        self.vae_decoder = VariationalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=self.embedding_dim
        ).to(device)
        
        # Classification head (for training on seen classes)
        self.classifier = nn.Linear(latent_dim, num_classes).to(device)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from N(mu, sigma^2).
        Equation 4 in paper: zV = mu + sigma ⊙ epsilon
        
        Args:
            mu: Mean [batch, dc]
            logvar: Log-variance [batch, dc]
            
        Returns:
            zV: Sampled variance features [batch, dc]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode input texts to embeddings using PubMedBERT.
        
        Args:
            texts: List of input text strings
            
        Returns:
            embeddings: Text embeddings [batch, H]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get [CLS] token embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, H]
            
        return embeddings
    
    def forward(
        self,
        texts: List[str],
        spurious_concepts: Optional[List[List[str]]] = None,
        labels: Optional[torch.Tensor] = None,
        return_augmentations: bool = False,
        num_augmentations: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.
        
        Args:
            texts: Input text strings
            spurious_concepts: List of spurious concept lists for each text
            labels: Ground truth labels [batch, num_classes] for training
            return_augmentations: Whether to generate augmentations for contrastive loss
            num_augmentations: Number of augmentations per sample (K in paper)
            
        Returns:
            Dictionary containing:
                - zI: Content features
                - zV: Variance features (if training)
                - z_reconstructed: Reconstructed embeddings
                - logits: Classification logits
                - mu, logvar: VAE parameters
                - augmentations: Augmented embeddings (if requested)
        """
        batch_size = len(texts)
        
        # Step 1: Encode texts to embeddings
        z_prime = self.encode_text(texts)  # [batch, H]
        
        # Step 2: Compute spurious directions and debias
        z_debiased_list = []
        
        for i in range(batch_size):
            if spurious_concepts is not None and spurious_concepts[i]:
                # Compute spurious direction for this instance
                vspur = self.spurious_estimator(spurious_concepts[i], self.tokenizer)
                
                # Apply debiasing
                z_debiased_i = self.debiaser(z_prime[i:i+1], vspur)
            else:
                # No spurious concepts, no debiasing needed
                z_debiased_i = z_prime[i:i+1]
                
            z_debiased_list.append(z_debiased_i)
        
        z = torch.cat(z_debiased_list, dim=0)  # [batch, H]
        
        # Step 3: Variational disentanglement
        zI, mu, logvar = self.vae_encoder(z)  # [batch, dc], [batch, dc], [batch, dc]
        
        # Sample variance features
        zV = self.reparameterize(mu, logvar)  # [batch, dc]
        
        # Combine for reconstruction
        z_combined = zI + zV  # [batch, dc]
        z_reconstructed = self.vae_decoder(z_combined)  # [batch, H]
        
        # Step 4: Classification (for training)
        logits = self.classifier(zI)  # [batch, num_classes]
        
        # Step 5: Generate augmentations for contrastive learning
        augmentations = None
        if return_augmentations:
            augmentations = []
            for k in range(num_augmentations):
                # Sample different variance features
                zV_k = self.reparameterize(mu, logvar)  # [batch, dc]
                z_aug_k = zI + zV_k  # [batch, dc]
                augmentations.append(z_aug_k)
            augmentations = torch.stack(augmentations, dim=1)  # [batch, K, dc]
        
        return {
            'zI': zI,
            'zV': zV,
            'z_reconstructed': z_reconstructed,
            'z_debiased': z,
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'augmentations': augmentations
        }
    
    def encode_labels(self, label_texts: List[str]) -> torch.Tensor:
        """
        Encode label descriptors to prototype embeddings.
        Used for zero-shot inference (Section 3.7).
        
        Args:
            label_texts: List of label descriptor texts
            
        Returns:
            label_prototypes: Normalized label embeddings [num_labels, H]
        """
        # Tokenize
        inputs = self.tokenizer(
            label_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get [CLS] token embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [num_labels, H]
        
        # Normalize
        label_prototypes = F.normalize(embeddings, p=2, dim=1)
        
        return label_prototypes


class ContrastiveLoss(nn.Module):
    """
    Spurious-Invariant Contrastive Regularization.
    Section 3.5 in paper, Equation 8.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        augmentations: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss over augmented embeddings.
        
        Args:
            augmentations: Augmented embeddings [batch, K, dc]
            labels: Class labels [batch]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size, K, dc = augmentations.shape
        
        # Reshape to [batch*K, dc]
        augmentations_flat = augmentations.reshape(-1, dc)
        
        # Normalize embeddings
        augmentations_norm = F.normalize(augmentations_flat, p=2, dim=1)
        
        # Compute similarity matrix: [batch*K, batch*K]
        sim_matrix = torch.matmul(augmentations_norm, augmentations_norm.T) / self.temperature
        
        # Create label matrix
        labels_expanded = labels.unsqueeze(1).expand(-1, K).reshape(-1)  # [batch*K]
        
        # Create mask for positive pairs (same class, different augmentation)
        labels_eq = labels_expanded.unsqueeze(0) == labels_expanded.unsqueeze(1)  # [batch*K, batch*K]
        
        # Exclude self-pairs
        mask_self = torch.eye(batch_size * K, dtype=torch.bool, device=augmentations.device)
        labels_eq = labels_eq & ~mask_self
        
        # Compute log probabilities
        exp_sim = torch.exp(sim_matrix)
        
        # Sum over all augmentations (denominator)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
        
        # Compute loss
        log_prob = sim_matrix - torch.log(sum_exp_sim + 1e-8)
        
        # Average over positive pairs
        loss = -(log_prob * labels_eq).sum(dim=1) / (labels_eq.sum(dim=1) + 1e-8)
        loss = loss.mean()
        
        return loss


def compute_reconstruction_loss(z: torch.Tensor, z_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Compute reconstruction loss.
    Equation 5 in paper: Lrec = ||zi - fdec(zI + zV)||^2
    
    Args:
        z: Original debiased embedding [batch, H]
        z_reconstructed: Reconstructed embedding [batch, H]
        
    Returns:
        loss: MSE reconstruction loss
    """
    return F.mse_loss(z_reconstructed, z)


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(zV|zi) and N(0, I).
    Part of Equation 6 in paper: DKL(q(zV|zi) || N(0, I))
    
    Args:
        mu: Mean of variational distribution [batch, dc]
        logvar: Log-variance of variational distribution [batch, dc]
        
    Returns:
        kl_loss: KL divergence loss
    """
    # KL[N(mu, sigma^2) || N(0, 1)] = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_loss.mean()
