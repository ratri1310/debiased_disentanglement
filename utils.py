"""
Utility functions for data preprocessing and dataset preparation.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_train_val_test_split(
    database_json: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = './data_splits',
    seed: int = 42
):
    """
    Split database into train/val/test sets and save PMIDs.
    
    Args:
        database_json: Path to database JSON
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        output_dir: Directory to save split files
        seed: Random seed
    """
    import numpy as np
    np.random.seed(seed)
    
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
    
    # Extract PMIDs
    pmids = [str(article['pmid']) for article in data]
    
    # Shuffle
    indices = np.arange(len(pmids))
    np.random.shuffle(indices)
    
    # Split
    n_train = int(len(pmids) * train_ratio)
    n_val = int(len(pmids) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_pmids = [pmids[i] for i in train_indices]
    val_pmids = [pmids[i] for i in val_indices]
    test_pmids = [pmids[i] for i in test_indices]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    with open(output_path / 'train_pmids.txt', 'w') as f:
        f.write('\n'.join(train_pmids))
    
    with open(output_path / 'val_pmids.txt', 'w') as f:
        f.write('\n'.join(val_pmids))
    
    with open(output_path / 'test_pmids.txt', 'w') as f:
        f.write('\n'.join(test_pmids))
    
    logger.info(f"Split data: {len(train_pmids)} train, {len(val_pmids)} val, {len(test_pmids)} test")
    logger.info(f"Saved to {output_path}")


def extract_unseen_labels(
    kg_output_dir: str,
    train_pmids_file: str,
    output_csv: str = 'unseen_labels.csv'
):
    """
    Extract labels that appear in test but not in training (unseen labels).
    
    Args:
        kg_output_dir: Directory with KG outputs
        train_pmids_file: File with training PMIDs
        output_csv: Output CSV file path
    """
    kg_dir = Path(kg_output_dir)
    
    # Load spurious detection results
    with open(kg_dir / 'spurious_detection.json', 'r') as f:
        spurious_data = json.load(f)
    
    # Load training PMIDs
    with open(train_pmids_file, 'r') as f:
        train_pmids = set(line.strip() for line in f)
    
    # Load nodes
    nodes_df = pd.read_csv(kg_dir / 'nodes.csv')
    
    # Collect labels from training set
    train_labels = set()
    for pmid in train_pmids:
        if pmid in spurious_data:
            train_labels.update(spurious_data[pmid]['total_mesh'])
    
    # Find unseen labels
    all_labels = set(nodes_df['mesh_id'].unique())
    unseen_labels = all_labels - train_labels
    
    # Create dataframe
    unseen_df = nodes_df[nodes_df['mesh_id'].isin(unseen_labels)][['mesh_id', 'pref_name']]
    unseen_df.columns = ['label_id', 'descriptor_text']
    
    # Save
    unseen_df.to_csv(output_csv, index=False)
    
    logger.info(f"Found {len(unseen_labels)} unseen labels")
    logger.info(f"Saved to {output_csv}")


def analyze_spurious_statistics(kg_output_dir: str):
    """
    Analyze and print statistics about spurious correlations.
    
    Args:
        kg_output_dir: Directory with KG outputs
    """
    kg_dir = Path(kg_output_dir)
    
    # Load spurious detection results
    with open(kg_dir / 'spurious_detection.json', 'r') as f:
        spurious_data = json.load(f)
    
    total_abstracts = len(spurious_data)
    total_concepts = sum(r['total_mesh'] for r in spurious_data.values())
    total_spurious = sum(r['spurious_mesh'] for r in spurious_data.values())
    total_connected = sum(r['connected_mesh'] for r in spurious_data.values())
    
    abstracts_with_spurious = sum(1 for r in spurious_data.values() if r['spurious_mesh'] > 0)
    
    # Get unique spurious codes
    all_spurious_codes = set()
    for result in spurious_data.values():
        all_spurious_codes.update(result['spurious_codes'])
    
    print("\n" + "="*60)
    print("SPURIOUS CORRELATION STATISTICS")
    print("="*60)
    print(f"Total abstracts: {total_abstracts}")
    print(f"Total MeSH concepts: {total_concepts}")
    print(f"Connected concepts: {total_connected} ({100*total_connected/total_concepts:.1f}%)")
    print(f"Spurious concepts: {total_spurious} ({100*total_spurious/total_concepts:.1f}%)")
    print(f"Unique spurious codes: {len(all_spurious_codes)}")
    print(f"Abstracts with spurious codes: {abstracts_with_spurious} ({100*abstracts_with_spurious/total_abstracts:.1f}%)")
    print("="*60 + "\n")
    
    # Load nodes to get names
    nodes_df = pd.read_csv(kg_dir / 'nodes.csv')
    mesh_to_name = dict(zip(nodes_df['mesh_id'], nodes_df['pref_name']))
    
    # Count frequency of each spurious code
    spurious_freq = {}
    for result in spurious_data.values():
        for code in result['spurious_codes']:
            spurious_freq[code] = spurious_freq.get(code, 0) + 1
    
    # Sort by frequency
    sorted_spurious = sorted(spurious_freq.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 20 Most Frequent Spurious Codes:")
    print("-" * 60)
    for i, (code, freq) in enumerate(sorted_spurious[:20], 1):
        name = mesh_to_name.get(code, code)
        print(f"{i:2d}. {code:10s} ({name:30s}): {freq:5d} ({100*freq/total_abstracts:.1f}%)")
    print()


def create_label_frequency_analysis(
    kg_output_dir: str,
    output_file: str = 'label_frequency_analysis.csv'
):
    """
    Analyze label frequency distribution for identifying tail labels.
    
    Args:
        kg_output_dir: Directory with KG outputs
        output_file: Output CSV file path
    """
    kg_dir = Path(kg_output_dir)
    
    # Load spurious detection
    with open(kg_dir / 'spurious_detection.json', 'r') as f:
        spurious_data = json.load(f)
    
    # Count label frequencies
    label_freq = {}
    for result in spurious_data.values():
        for code in result['total_mesh']:
            label_freq[code] = label_freq.get(code, 0) + 1
    
    # Load nodes
    nodes_df = pd.read_csv(kg_dir / 'nodes.csv')
    mesh_to_name = dict(zip(nodes_df['mesh_id'], nodes_df['pref_name']))
    
    # Create dataframe
    freq_data = []
    for code, freq in label_freq.items():
        freq_data.append({
            'mesh_id': code,
            'name': mesh_to_name.get(code, code),
            'frequency': freq,
            'percentage': 100 * freq / len(spurious_data)
        })
    
    freq_df = pd.DataFrame(freq_data)
    freq_df = freq_df.sort_values('frequency', ascending=False)
    
    # Add tail label indicator (< 5% frequency)
    freq_df['is_tail'] = freq_df['percentage'] < 5.0
    
    # Save
    freq_df.to_csv(output_file, index=False)
    
    # Print summary
    tail_labels = freq_df['is_tail'].sum()
    print(f"\nLabel Frequency Analysis:")
    print(f"Total labels: {len(freq_df)}")
    print(f"Tail labels (<5% frequency): {tail_labels} ({100*tail_labels/len(freq_df):.1f}%)")
    print(f"Head labels (≥5% frequency): {len(freq_df) - tail_labels} ({100*(len(freq_df)-tail_labels)/len(freq_df):.1f}%)")
    print(f"Saved to {output_file}\n")


def verify_data_integrity(
    database_json: str,
    kg_output_dir: str
):
    """
    Verify data integrity between database and KG outputs.
    
    Args:
        database_json: Path to database JSON
        kg_output_dir: Directory with KG outputs
    """
    kg_dir = Path(kg_output_dir)
    
    print("\nVerifying data integrity...")
    print("-" * 60)
    
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
    
    db_pmids = set(str(article['pmid']) for article in data)
    print(f"✓ Database PMIDs: {len(db_pmids)}")
    
    # Load spurious detection
    with open(kg_dir / 'spurious_detection.json', 'r') as f:
        spurious_data = json.load(f)
    
    kg_pmids = set(spurious_data.keys())
    print(f"✓ KG PMIDs: {len(kg_pmids)}")
    
    # Check overlap
    common_pmids = db_pmids & kg_pmids
    print(f"✓ Common PMIDs: {len(common_pmids)} ({100*len(common_pmids)/len(db_pmids):.1f}%)")
    
    if len(common_pmids) < len(db_pmids):
        missing = len(db_pmids) - len(common_pmids)
        print(f"⚠ Missing {missing} PMIDs in KG outputs")
    
    # Check nodes file
    nodes_df = pd.read_csv(kg_dir / 'nodes.csv')
    print(f"✓ Graph nodes: {len(nodes_df)}")
    
    # Check edges file
    edges_df = pd.read_csv(kg_dir / 'edges.csv')
    print(f"✓ Graph edges: {len(edges_df)}")
    
    # Check maps
    with open(kg_dir / 'maps.json', 'r') as f:
        maps = json.load(f)
    
    print(f"✓ MeSH to CUI mappings: {len(maps['mesh_to_cui'])}")
    print(f"✓ CUI to ID mappings: {len(maps['cui_to_id'])}")
    
    print("-" * 60)
    print("Data integrity check completed!\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preprocessing utilities')
    parser.add_argument('--command', type=str, required=True,
                       choices=['split', 'unseen', 'stats', 'freq', 'verify'],
                       help='Command to run')
    parser.add_argument('--database-json', type=str,
                       help='Path to database JSON')
    parser.add_argument('--kg-output-dir', type=str,
                       help='KG output directory')
    parser.add_argument('--output-dir', type=str, default='./data_splits',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        create_train_val_test_split(
            args.database_json,
            output_dir=args.output_dir,
            seed=args.seed
        )
    elif args.command == 'unseen':
        extract_unseen_labels(
            args.kg_output_dir,
            f"{args.output_dir}/train_pmids.txt",
            f"{args.output_dir}/unseen_labels.csv"
        )
    elif args.command == 'stats':
        analyze_spurious_statistics(args.kg_output_dir)
    elif args.command == 'freq':
        create_label_frequency_analysis(
            args.kg_output_dir,
            f"{args.output_dir}/label_frequency_analysis.csv"
        )
    elif args.command == 'verify':
        verify_data_integrity(args.database_json, args.kg_output_dir)
