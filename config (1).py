"""
Configuration file for training and inference.

Example usage:
    from config import Config
    config = Config()
    # Or load from file:
    config = Config.from_yaml('config.yaml')
"""

import yaml
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class for the debiased disentanglement model."""
    
    # Data paths
    umls_dir: str = '/path/to/UMLS/META'
    mesh_xlsx: str = '/path/to/meshcodes.xlsx'
    database_json: str = '/path/to/database.json'
    kg_output_dir: str = './kg_outputs'
    output_dir: str = './checkpoints'
    
    # Model architecture
    pretrained_model: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    hidden_dim: int = 512
    latent_dim: int = 256
    num_classes: int = 2574
    
    # Training hyperparameters
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Loss weights (Equation 9 in paper)
    gamma: float = 1.0          # Debiasing strength
    beta: float = 1.0           # KL divergence weight
    alpha: float = 1.0          # Classification loss weight
    lambda_contrast: float = 0.5  # Contrastive loss weight
    
    # Contrastive learning
    temperature: float = 0.07
    num_augmentations: int = 5
    
    # System
    device: str = 'cuda'
    num_workers: int = 4
    seed: int = 42
    
    # Inference
    top_k: int = 15
    use_faiss: bool = True
    
    def __init__(self, **kwargs):
        """Initialize config with optional keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith('_') and not callable(getattr(self, key))
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self):
        """String representation of config."""
        lines = ["Configuration:"]
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                lines.append(f"  {key}: {getattr(self, key)}")
        return '\n'.join(lines)


# Example configuration templates

NEUROLOGY_CONFIG = Config(
    database_json='/path/to/neurology_database.json',
    kg_output_dir='./kg_outputs/neurology',
    output_dir='./checkpoints/neurology',
    num_classes=2574,
    batch_size=16,
    epochs=50
)

IMMUNOLOGY_CONFIG = Config(
    database_json='/path/to/immunology_database.json',
    kg_output_dir='./kg_outputs/immunology',
    output_dir='./checkpoints/immunology',
    num_classes=9333,
    batch_size=16,
    epochs=50
)

EMBRYOLOGY_CONFIG = Config(
    database_json='/path/to/embryology_database.json',
    kg_output_dir='./kg_outputs/embryology',
    output_dir='./checkpoints/embryology',
    num_classes=14135,
    batch_size=16,
    epochs=50
)


if __name__ == '__main__':
    # Example: Create and save configuration
    config = Config()
    print(config)
    
    # Save to YAML
    config.to_yaml('config.yaml')
    print("\nSaved configuration to config.yaml")
    
    # Load from YAML
    loaded_config = Config.from_yaml('config.yaml')
    print("\nLoaded configuration:")
    print(loaded_config)
