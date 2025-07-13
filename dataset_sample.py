#!/usr/bin/env python3
"""
Sample Dataset Generator Script

This script generates cluster-based sample datasets and metadata using the SampledATEBaselines class.
It allows users to specify custom sample ratios and generates representative subsets of large datasets.

Usage:
    python generate_sample.py [--data_file DATA_FILE] [--sample_ratio RATIO] [--output_prefix PREFIX]

Requirements:
    - sampled_top_k.py (contains SampledATEBaselines class)
    - clustered_top_k.py (dependency of SampledATEBaselines)
    - ATE_update.py (dependency of SampledATEBaselines)
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Import the required class (ensure the file is in the same directory)
try:
    from sampled_top_k import SampledATEBaselines
except ImportError as e:
    print(f"Error: Could not import SampledATEBaselines. Make sure sampled_top_k.py is in the same directory.")
    print(f"Import error: {e}")
    sys.exit(1)


def load_data_from_csv(file_path, feature_cols=None, treatment_col='T', outcome_col='Y'):
    """
    Load data from CSV file and extract features, treatment, and outcome.
    
    Args:
        file_path: Path to CSV file
        feature_cols: List of feature column names (if None, auto-detect)
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        
    Returns:
        tuple: (X, T, Y) DataFrames/Series
    """
    print(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")
    
    # Check if required columns exist
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found in dataset")
    
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in dataset")
    
    # Extract treatment and outcome
    T = df[treatment_col]
    Y = df[outcome_col]
    
    # Extract features
    if feature_cols is None:
        # Auto-detect feature columns (all columns except treatment and outcome)
        feature_cols = [col for col in df.columns if col not in [treatment_col, outcome_col]]
        print(f"Auto-detected feature columns: {feature_cols}")
    else:
        # Validate specified feature columns
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
    
    X = df[feature_cols]
    
    print(f"Features shape: {X.shape}")
    print(f"Treatment unique values: {T.unique()}")
    print(f"Outcome statistics: mean={Y.mean():.4f}, std={Y.std():.4f}")
    
    return X, T, Y


def validate_sample_ratio(ratio):
    """Validate that sample ratio is within reasonable bounds."""
    if not 0 < ratio <= 1:
        raise ValueError(f"Sample ratio must be between 0 and 1, got {ratio}")
    
    if ratio < 0.001:
        print(f"Warning: Very small sample ratio ({ratio}). Consider using at least 0.001 (0.1%)")
    
    if ratio > 0.5:
        print(f"Warning: Large sample ratio ({ratio}). Consider using smaller ratios for efficiency")
    
    return ratio


def generate_sample_dataset(X, T, Y, sample_ratio, output_prefix="sample", 
                          model_type='linear', random_seed=42, force_resample=True):
    """
    Generate sample dataset using cluster-based sampling.
    
    Args:
        X, T, Y: Feature matrix, treatment vector, outcome vector
        sample_ratio: Proportion of data to sample (0 < ratio <= 1)
        output_prefix: Prefix for output files
        model_type: Model type ('linear' or 'logistic')
        random_seed: Random seed for reproducibility
        force_resample: Whether to force new sampling (ignore cached files)
        
    Returns:
        dict: Information about the generated sample
    """
    
    print(f"\n{'='*60}")
    print(f"GENERATING CLUSTER-BASED SAMPLE DATASET")
    print(f"{'='*60}")
    
    # Validate inputs
    sample_ratio = validate_sample_ratio(sample_ratio)
    
    # Create output file paths
    sample_file = f"{output_prefix}_ratio_{sample_ratio:.3f}.csv"
    
    print(f"Sample ratio: {sample_ratio:.1%}")
    print(f"Target sample size: ~{int(len(X) * sample_ratio)} points")
    print(f"Output file: {sample_file}")
    print(f"Random seed: {random_seed}")
    print(f"Model type: {model_type}")
    
    # Create SampledATEBaselines instance
    try:
        sampler = SampledATEBaselines(
            X=X,
            T=T, 
            Y=Y,
            model_type=model_type,
            sample_ratio=sample_ratio,
            random_seed=random_seed,
            sample_file_path=sample_file,
            use_cached_sample=not force_resample,
            save_sample=True,
            force_resample=force_resample,
            log_dir='logs'
        )
        
        print(f"\n✓ Sample dataset generated successfully!")
        
    except Exception as e:
        print(f"\n✗ Error generating sample: {e}")
        raise
    
    # Get sample information
    sample_info = sampler.get_sample_info()
    
    # Get file information
    file_info = sampler.get_sample_file_info()
    
    # Display results
    print(f"\n{'='*40}")
    print(f"SAMPLE GENERATION RESULTS")
    print(f"{'='*40}")
    
    print(f"Original dataset size: {sample_info['full_dataset_size']:,}")
    print(f"Sample dataset size: {sample_info['sample_size']:,}")
    print(f"Actual sample ratio: {sample_info['sample_size']/sample_info['full_dataset_size']:.3%}")
    print(f"Sampling method: {sample_info['sampling_method']}")
    
    if sample_info.get('n_clusters_used'):
        print(f"Clusters used: {sample_info['n_clusters_used']}")
        
        if 'cluster_distribution' in sample_info:
            cluster_dist = sample_info['cluster_distribution']
            print(f"Cluster size range: {cluster_dist['min_cluster_size']}-{cluster_dist['max_cluster_size']}")
            print(f"Average cluster size: {cluster_dist['avg_cluster_size']:.1f}")
    
    print(f"\n{'='*40}")
    print(f"OUTPUT FILES")
    print(f"{'='*40}")
    
    # CSV file info
    if file_info['sample_file']['exists']:
        csv_info = file_info['sample_file']
        print(f"✓ CSV file: {csv_info['path']}")
        print(f"  Size: {csv_info['file_size_mb']:.2f} MB")
        print(f"  Columns: {len(csv_info['columns'])}")
        print(f"  Sample size: {csv_info['sample_size']:,}")
    
    # Metadata file info  
    if file_info['metadata_file']['exists']:
        meta_info = file_info['metadata_file']
        print(f"✓ Metadata file: {meta_info['path']}")
        print(f"  Size: {meta_info['file_size_mb']:.4f} MB")
        
        if 'metadata' in meta_info:
            metadata = meta_info['metadata']
            print(f"  Created: {metadata.get('created_timestamp', 'Unknown')}")
            print(f"  Random seed: {metadata.get('random_seed', 'Unknown')}")
    
    # Return comprehensive information
    result = {
        'sampler': sampler,
        'sample_info': sample_info,
        'file_info': file_info,
        'files': {
            'csv': sample_file,
            'metadata': sampler._get_metadata_file_path()
        }
    }
    
    return result


def interactive_mode():
    """Run the script in interactive mode."""
    print(f"\n{'='*60}")
    print(f"INTERACTIVE SAMPLE DATASET GENERATOR")
    print(f"{'='*60}")
    
    # Get data file path
    while True:
        data_file = input("\nEnter path to CSV data file: ").strip()
        if os.path.exists(data_file):
            break
        print(f"File not found: {data_file}")
    
    # Load and inspect data
    try:
        df = pd.read_csv(data_file)
        print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get column specifications
    print(f"\nSpecify column roles:")
    
    treatment_col = input(f"Treatment column name (default: 'T'): ").strip() or 'T'
    outcome_col = input(f"Outcome column name (default: 'Y'): ").strip() or 'Y'
    
    feature_cols_input = input(f"Feature columns (comma-separated, leave empty for auto-detect): ").strip()
    feature_cols = [col.strip() for col in feature_cols_input.split(',')] if feature_cols_input else None
    
    # Load data with specifications
    try:
        X, T, Y = load_data_from_csv(data_file, feature_cols, treatment_col, outcome_col)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # Get sampling parameters
    while True:
        try:
            ratio_input = input(f"\nSample ratio (0-1, e.g., 0.1 for 10%): ").strip()
            sample_ratio = float(ratio_input)
            validate_sample_ratio(sample_ratio)
            break
        except ValueError as e:
            print(f"Invalid ratio: {e}")
    
    output_prefix = input(f"Output file prefix (default: 'sample'): ").strip() or 'sample'
    
    model_type = input(f"Model type (linear/logistic, default: 'linear'): ").strip() or 'linear'
    
    random_seed_input = input(f"Random seed (default: 42): ").strip()
    random_seed = int(random_seed_input) if random_seed_input else 42
    
    # Generate sample
    try:
        result = generate_sample_dataset(
            X, T, Y, sample_ratio, output_prefix, model_type, random_seed
        )
        
        print(f"\n✓ Sample generation completed successfully!")
        print(f"Files saved:")
        print(f"  - {result['files']['csv']}")
        print(f"  - {result['files']['metadata']}")
        
    except Exception as e:
        print(f"\n✗ Sample generation failed: {e}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate cluster-based sample datasets with metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python generate_sample.py
    
    # Generate 10% sample from data.csv
    python generate_sample.py --data_file data.csv --sample_ratio 0.1
    
    # Generate 5% sample with custom output prefix
    python generate_sample.py --data_file data.csv --sample_ratio 0.05 --output_prefix my_sample
    
    # Specify custom column names
    python generate_sample.py --data_file data.csv --sample_ratio 0.1 --treatment_col treatment --outcome_col outcome
        """
    )
    
    parser.add_argument('--data_file', type=str, default="acs_encoded.csv", help='Path to CSV data file')
    parser.add_argument('--sample_ratio', type=float, help='Sample ratio (0-1)')
    parser.add_argument('--output_prefix', type=str, default='acs_sample_ratio', help='Output file prefix')
    parser.add_argument('--model_type', type=str, choices=['linear', 'logistic'], default='linear', help='Model type')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force_resample', action='store_true', help='Force new sampling (ignore cached files)')
    
    args = parser.parse_args()
    
    # If no arguments provided, run interactive mode
    if not any(vars(args).values()) or (args.data_file is None and args.sample_ratio is None):
        interactive_mode()
        return
    
    # Validate required arguments for non-interactive mode
    if args.data_file is None:
        parser.error("--data_file is required in non-interactive mode")
    
    if args.sample_ratio is None:
        parser.error("--sample_ratio is required in non-interactive mode")
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    treatment = "With a disability"
    outcome = "Wages or salary income past 12 months"
    features_cols = ["Educational attainment", "Public health coverage", "Private health insurance coverage", "Medicare, for people 65 and older, or people with certain disabilities", "Insurance through a current or former employer or union", "Sex", "Age"]

    # Load data
    try:
        X, T, Y = load_data_from_csv(
            args.data_file, features_cols, treatment, outcome
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Generate sample
    try:
        result = generate_sample_dataset(
            X, T, Y, 
            args.sample_ratio, 
            args.output_prefix,
            args.model_type,
            args.random_seed,
            args.force_resample
        )
        
        print(f"\n✓ Sample generation completed successfully!")
        print(f"Files saved:")
        print(f"  - {result['files']['csv']}")
        print(f"  - {result['files']['metadata']}")
        
    except Exception as e:
        print(f"\n✗ Sample generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
