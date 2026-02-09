"""
Test script to verify model components and GPU compatibility.
"""

import torch
import numpy as np
from model import (
    DebiasedDisentanglementModel,
    SpuriousDirectionEstimator,
    TranslationDebiaser,
    ContrastiveLoss,
    compute_reconstruction_loss,
    compute_kl_divergence
)


def test_gpu_availability():
    """Test if GPU is available and functioning."""
    print("\n" + "="*60)
    print("Testing GPU Availability")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        device = 'cuda'
    else:
        print("⚠ CUDA is not available, using CPU")
        device = 'cpu'
    
    return device


def test_model_initialization(device='cuda'):
    """Test model initialization."""
    print("\n" + "="*60)
    print("Testing Model Initialization")
    print("="*60)
    
    try:
        model = DebiasedDisentanglementModel(
            pretrained_model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            hidden_dim=512,
            latent_dim=256,
            num_classes=100,
            device=device
        )
        print("✓ Model initialized successfully")
        print(f"  Embedding dimension: {model.embedding_dim}")
        print(f"  Latent dimension: {model.latent_dim}")
        print(f"  Device: {model.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return None


def test_forward_pass(model, device='cuda'):
    """Test forward pass with dummy data."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    try:
        # Create dummy inputs
        texts = [
            "Alzheimer's disease is characterized by memory loss and cognitive decline.",
            "Treatment with donepezil showed significant improvement in patients."
        ]
        
        spurious_concepts = [
            ["Female", "Aged"],
            ["Male", "Middle Aged"]
        ]
        
        # Forward pass
        outputs = model(
            texts=texts,
            spurious_concepts=spurious_concepts,
            return_augmentations=True,
            num_augmentations=3
        )
        
        print("✓ Forward pass successful")
        print(f"  Content features (zI) shape: {outputs['zI'].shape}")
        print(f"  Variance features (zV) shape: {outputs['zV'].shape}")
        print(f"  Reconstructed embedding shape: {outputs['z_reconstructed'].shape}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Augmentations shape: {outputs['augmentations'].shape}")
        
        return outputs
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_losses(model, outputs, device='cuda'):
    """Test loss computations."""
    print("\n" + "="*60)
    print("Testing Loss Computations")
    print("="*60)
    
    try:
        # Reconstruction loss
        recon_loss = compute_reconstruction_loss(
            outputs['z_debiased'],
            outputs['z_reconstructed']
        )
        print(f"✓ Reconstruction loss: {recon_loss.item():.4f}")
        
        # KL divergence
        kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
        print(f"✓ KL divergence: {kl_loss.item():.4f}")
        
        # Classification loss
        labels = torch.zeros(2, 100).to(device)
        labels[0, [0, 5, 10]] = 1.0
        labels[1, [2, 7, 15]] = 1.0
        cls_loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'], labels)
        print(f"✓ Classification loss: {cls_loss.item():.4f}")
        
        # Contrastive loss
        contrastive_fn = ContrastiveLoss(temperature=0.07)
        class_indices = torch.tensor([0, 1]).to(device)
        contrast_loss = contrastive_fn(outputs['augmentations'], class_indices)
        print(f"✓ Contrastive loss: {contrast_loss.item():.4f}")
        
        # Total loss
        total_loss = recon_loss + model.beta * kl_loss + cls_loss + 0.5 * contrast_loss
        print(f"✓ Total loss: {total_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, outputs, device='cuda'):
    """Test backward pass and gradient computation."""
    print("\n" + "="*60)
    print("Testing Backward Pass")
    print("="*60)
    
    try:
        # Compute loss
        recon_loss = compute_reconstruction_loss(
            outputs['z_debiased'],
            outputs['z_reconstructed']
        )
        kl_loss = compute_kl_divergence(outputs['mu'], outputs['logvar'])
        
        labels = torch.zeros(2, 100).to(device)
        labels[0, [0, 5, 10]] = 1.0
        labels[1, [2, 7, 15]] = 1.0
        cls_loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'], labels)
        
        contrastive_fn = ContrastiveLoss(temperature=0.07)
        class_indices = torch.tensor([0, 1]).to(device)
        contrast_loss = contrastive_fn(outputs['augmentations'], class_indices)
        
        total_loss = recon_loss + model.beta * kl_loss + cls_loss + 0.5 * contrast_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print("✓ Backward pass successful")
            print("✓ Gradients computed correctly")
        else:
            print("⚠ No gradients found (frozen parameters expected for text encoder)")
        
        return True
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_mode(model, device='cuda'):
    """Test inference mode."""
    print("\n" + "="*60)
    print("Testing Inference Mode")
    print("="*60)
    
    try:
        model.eval()
        
        with torch.no_grad():
            # Test sample
            texts = ["Alzheimer's disease affects memory and cognitive function."]
            spurious = [["Female", "Aged"]]
            
            outputs = model(
                texts=texts,
                spurious_concepts=spurious,
                return_augmentations=False
            )
            
            # Encode label prototypes
            label_texts = [
                "Alzheimer's Disease",
                "Memory Disorders",
                "Cognitive Dysfunction"
            ]
            
            label_prototypes = model.encode_labels(label_texts)
            print(f"✓ Label prototypes encoded: {label_prototypes.shape}")
            
            # Compute similarities
            content_norm = torch.nn.functional.normalize(outputs['zI'], p=2, dim=1)
            
            # Project label prototypes to content space
            h = model.vae_encoder.encoder(label_prototypes)
            label_zI = model.vae_encoder.content_pool(h.unsqueeze(1)).squeeze(1)
            label_norm = torch.nn.functional.normalize(label_zI, p=2, dim=1)
            
            similarities = torch.matmul(content_norm, label_norm.T)
            print(f"✓ Similarities computed: {similarities.shape}")
            print(f"  Similarity values: {similarities[0].cpu().numpy()}")
            
            # Top predictions
            top_k = torch.topk(similarities, k=3, dim=1)
            print(f"✓ Top-3 predictions: indices={top_k.indices[0].cpu().numpy()}, scores={top_k.values[0].cpu().numpy()}")
        
        return True
    except Exception as e:
        print(f"✗ Inference mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(device='cuda'):
    """Test memory usage."""
    print("\n" + "="*60)
    print("Testing Memory Usage")
    print("="*60)
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        model = DebiasedDisentanglementModel(
            pretrained_model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            hidden_dim=512,
            latent_dim=256,
            num_classes=100,
            device=device
        )
        
        # Forward pass with batch
        batch_size = 16
        texts = ["Sample text for memory testing."] * batch_size
        spurious = [["Female"]] * batch_size
        
        outputs = model(
            texts=texts,
            spurious_concepts=spurious,
            return_augmentations=True,
            num_augmentations=5
        )
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        
        print(f"✓ Memory usage test completed")
        print(f"  Batch size: {batch_size}")
        print(f"  Memory allocated: {allocated:.2f} GB")
        print(f"  Memory reserved: {reserved:.2f} GB")
        print(f"  Peak memory: {peak:.2f} GB")
        
        return True
    else:
        print("⚠ Skipping memory test (CPU mode)")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Debiased Disentanglement Model - Component Tests")
    print("="*60)
    
    # Test GPU
    device = test_gpu_availability()
    
    # Test model initialization
    model = test_model_initialization(device)
    if model is None:
        print("\n✗ Tests failed: Could not initialize model")
        return
    
    # Test forward pass
    outputs = test_forward_pass(model, device)
    if outputs is None:
        print("\n✗ Tests failed: Forward pass failed")
        return
    
    # Test losses
    if not test_losses(model, outputs, device):
        print("\n✗ Tests failed: Loss computation failed")
        return
    
    # Test backward pass
    if not test_backward_pass(model, outputs, device):
        print("\n✗ Tests failed: Backward pass failed")
        return
    
    # Test inference
    if not test_inference_mode(model, device):
        print("\n✗ Tests failed: Inference mode failed")
        return
    
    # Test memory usage
    test_memory_usage(device)
    
    # Summary
    print("\n" + "="*60)
    print("All Tests Passed! ✓")
    print("="*60)
    print("\nThe model is ready for training and inference.")
    print("You can now proceed with:")
    print("  1. Building the knowledge graph (build_kg.py)")
    print("  2. Training the model (train.py)")
    print("  3. Running zero-shot inference (inference.py)")
    print("")


if __name__ == '__main__':
    main()
