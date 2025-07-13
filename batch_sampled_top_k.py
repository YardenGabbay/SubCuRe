import pandas as pd
import numpy as np
import os
import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sampled_top_k import SampledATEBaselines
from ATE_update import ATEUpdateLinear, ATEUpdateLogistic

class BatchSampledATEBaselines(SampledATEBaselines):
    """
    Enhanced Scalable implementation of Top-k and Top-k+ baseline methods for ATE modification
    with batch removal strategy.
    
    This class extends SampledATEBaselines with a more precise batch removal approach:
    1. Cluster the full dataset into representative groups using K-means on (X, T, Y)
    2. Sample proportionally from each cluster to create a representative subset
    3. Run clustered top-k/top-k+ on the representative subset
    4. For each removed point from the subset, find k-nearest neighbors in the full dataset
    5. **NEW**: Remove neighbors in batches (5 at a time) until target ATE is reached or all are removed
    
    This approach provides more precise control over the final ATE while maintaining computational efficiency.
    
    New features (inherited from SampledATEBaselines):
    - Save sampled subset to CSV file for reuse
    - Load previously sampled subset from CSV file
    - Automatic caching and validation of sample files
    - **NEW**: Log verbose output to text files for analysis
    """
    
    def __init__(self, X, T, Y, model_type='linear', lambda_reg=0.1, max_iter=1000, find_confounders=False, sample_ratio=0.1, random_seed=42, batch_size=5, sample_file_path='acs_sample.csv', use_cached_sample=True, save_sample=True, force_resample=False, log_dir='logs'):
        """
        Initialize the BatchSampledATEBaselines with cluster-based sampling, caching, batch removal, and logging.
        
        Args:
            X, T, Y: Full dataset features, treatment, and outcome
            model_type: 'linear' or 'logistic'
            lambda_reg: Regularization parameter for logistic regression
            max_iter: Maximum iterations for logistic regression
            find_confounders: Whether to find confounders automatically
            sample_ratio: Ratio of data to sample using cluster-based method (default 0.1 for 10%)
            random_seed: Random seed for reproducible clustering and sampling
            batch_size: Number of points to remove in each batch (default 5)
            sample_file_path: Path to save/load the sample CSV file (default 'acs_sample.csv')
            use_cached_sample: Whether to try loading existing sample file first (default True)
            save_sample: Whether to save the sample to CSV after creating it (default True)
            force_resample: If True, ignore existing sample file and create new sample (default False)
            log_dir: Directory to save log files (default 'logs')
        """
        # Initialize parent class with caching parameters
        super().__init__(X, T, Y, model_type, lambda_reg, max_iter, 
                        find_confounders, sample_ratio, random_seed,
                        sample_file_path, use_cached_sample, save_sample, force_resample, log_dir)
        
        # Store batch removal parameters
        self.batch_size = batch_size
    
    def batch_sampled_top_k(self, target_ate=None, epsilon=0.01, k=None, k_neighbors=5, approx=False, use_clustering=True, n_clusters=None, samples_per_cluster=2, verbose=True, log_file=None):
        """
        Scalable Top-k method using cluster-based sampling and batch KNN removal.
        
        Args:
            target_ate: Target ATE value
            epsilon: Tolerance for target ATE
            k: Maximum number of points to remove (if target_ate is None)
            k_neighbors: Number of nearest neighbors to find for each removed point
            approx: Whether to use approximation for ATE calculation
            use_clustering: Whether to use clustering in the subset
            n_clusters: Number of clusters (if None, automatically determined)
            samples_per_cluster: Number of samples per cluster
            verbose: Whether to log progress information
            log_file: Custom log file name (if None, auto-generated)
            
        Returns:
            Dictionary containing results of the algorithm
        """
        # Setup logging
        if verbose:
            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"batch_sampled_top_k_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Batch Sampled Top-k Algorithm (Cluster-based Sampling) ===")
            self._log(f"Sample ratio: {self.sample_ratio*100}%")
            self._log(f"Sample size: {len(self.X_sample)}")
            self._log(f"Full dataset size: {len(self.X_full)}")
            self._log(f"Clusters used for sampling: {getattr(self, 'n_clusters_used', 'N/A')}")
            self._log(f"K-neighbors: {k_neighbors}")
            self._log(f"Batch size: {self.batch_size}")
            self._log(f"Sample file: {self.sample_file_path}")
            self._log(f"Log file: {self.current_log_file}")
        
        # Step 1: Run clustered top-k on cluster-based sampled subset
        if verbose:
            self._log(f"\n--- Step 1: Running clustered top-k on cluster-based sampled subset ---")
        
        sample_result = super().top_k(target_ate=target_ate, epsilon=epsilon, k=k, approx=approx, use_clustering=use_clustering, n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, verbose=verbose)
        
        # Step 2: Find k-nearest neighbors in full dataset
        if verbose:
            self._log(f"\n--- Step 2: Finding nearest neighbors in full dataset ---")
        
        knn_indices = self._find_knn_in_full_dataset(
            sample_result['removed_indices'], k_neighbors, verbose=verbose)
        
        if verbose:
            self._log(f"Found {len(knn_indices)} unique nearest neighbors to remove")
        
        # Step 3: Gradually remove KNN neighbors in batches until target is reached
        if verbose:
            self._log(f"\n--- Step 3: Batch removal of neighbors from full dataset ---")
        
        original_full_ate, final_ate, final_removed_indices = self._batch_remove_neighbors(
            knn_indices, target_ate, epsilon, approx, verbose)
        
        # Compile results
        result = {
            'removed_indices': final_removed_indices,
            'final_ate': final_ate,
            'num_removed': len(final_removed_indices),
            'original_ate': original_full_ate,
            'ate_difference': final_ate - original_full_ate,
            'sample_result': sample_result,
            'sample_size': len(self.X_sample),
            'full_dataset_size': len(self.X_full),
            'k_neighbors': k_neighbors,
            'sample_ratio': self.sample_ratio,
            'computational_efficiency': len(self.X_sample) / len(self.X_full),
            'sampling_method': 'cluster-based',
            'knn_candidates': len(knn_indices),  # Total candidates found by KNN
            'batch_removal': True,  # Indicates batch removal was used
            'batch_size': self.batch_size,
            'sample_file_path': self.sample_file_path,
            'log_file': self.current_log_file if verbose else None
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE (full dataset): {result['original_ate']:.6f}")
            self._log(f"Final ATE: {final_ate:.6f}")
            self._log(f"ATE Difference: {result['ate_difference']:.6f}")
            self._log(f"KNN candidates found: {len(knn_indices)}")
            self._log(f"Points actually removed: {len(final_removed_indices)}")
            self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result
    
    def batch_sampled_top_k_plus(self, target_ate=None, epsilon=0.01, k=None, k_neighbors=5, approx=False, use_clustering=True, n_clusters=None, samples_per_cluster=2, influence_recalc_interval=1, verbose=True, log_file=None):
        """
        Scalable Top-k+ method using cluster-based sampling and batch KNN removal.
        
        Args:
            target_ate: Target ATE value
            epsilon: Tolerance for target ATE
            k: Maximum number of points to remove (if target_ate is None)
            k_neighbors: Number of nearest neighbors to find for each removed point
            approx: Whether to use approximation for ATE calculation
            use_clustering: Whether to use clustering in the subset
            n_clusters: Number of clusters (if None, automatically determined)
            samples_per_cluster: Number of samples per cluster
            verbose: Whether to log progress information
            log_file: Custom log file name (if None, auto-generated)
            
        Returns:
            Dictionary containing results of the algorithm
        """
        # Setup logging
        if verbose:
            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"batch_sampled_top_k_plus_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Batch Sampled Top-k+ Algorithm (Cluster-based Sampling) ===")
            self._log(f"Sample ratio: {self.sample_ratio*100}%")
            self._log(f"Sample size: {len(self.X_sample)}")
            self._log(f"Full dataset size: {len(self.X_full)}")
            self._log(f"Clusters used for sampling: {getattr(self, 'n_clusters_used', 'N/A')}")
            self._log(f"K-neighbors: {k_neighbors}")
            self._log(f"Batch size: {self.batch_size}")
            self._log(f"Sample file: {self.sample_file_path}")
            self._log(f"Log file: {self.current_log_file}")
        
        # Step 1: Run clustered top-k+ on cluster-based sampled subset
        if verbose:
            self._log(f"\n--- Step 1: Running clustered top-k+ on cluster-based sampled subset ---")

        sample_result = super().top_k_plus_best(target_ate=target_ate, epsilon=epsilon, k=k, approx=approx, use_clustering=use_clustering, n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, influence_recalc_interval=influence_recalc_interval, verbose=verbose)

        # Step 2: Find k-nearest neighbors in full dataset
        if verbose:
            self._log(f"\n--- Step 2: Finding nearest neighbors in full dataset ---")
        
        knn_indices = self._find_knn_in_full_dataset(
            sample_result['removed_indices'], k_neighbors, verbose=verbose)
        
        if verbose:
            self._log(f"Found {len(knn_indices)} unique nearest neighbors to remove")
        
        # Step 3: Gradually remove KNN neighbors in batches until target is reached
        if verbose:
            self._log(f"\n--- Step 3: Batch removal of neighbors from full dataset ---")
        
        original_full_ate, final_ate, final_removed_indices = self._batch_remove_neighbors(
            knn_indices, target_ate, epsilon, approx, verbose)
        
        # Compile results
        result = {
            'removed_indices': final_removed_indices,
            'final_ate': final_ate,
            'num_removed': len(final_removed_indices),
            'original_ate': original_full_ate,
            'ate_difference': final_ate - original_full_ate,
            'sample_result': sample_result,
            'sample_size': len(self.X_sample),
            'full_dataset_size': len(self.X_full),
            'k_neighbors': k_neighbors,
            'sample_ratio': self.sample_ratio,
            'computational_efficiency': len(self.X_sample) / len(self.X_full),
            'sampling_method': 'cluster-based',
            'knn_candidates': len(knn_indices),  # Total candidates found by KNN
            'batch_removal': True,  # Indicates batch removal was used
            'batch_size': self.batch_size,
            'sample_file_path': self.sample_file_path,
            'log_file': self.current_log_file if verbose else None
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE (full dataset): {result['original_ate']:.6f}")
            self._log(f"Final ATE: {final_ate:.6f}")
            self._log(f"ATE Difference: {result['ate_difference']:.6f}")
            self._log(f"KNN candidates found: {len(knn_indices)}")
            self._log(f"Points actually removed: {len(final_removed_indices)}")
            self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result
    
    def _batch_remove_neighbors(self,
                             knn_indices,
                             target_ate,
                             epsilon,
                             approx,
                             verbose):
        """
        Remove KNN neighbors in batches *with index-shift compensation*.

        After every drop we assume the user will
        `reset_index(drop=True)`, so the remaining rows' positions change.
        This function therefore:
            1. sorts the candidate indices,
            2. keeps two parallel lists
            · removed_orig     – indices in the *original* coordinate system  
            · removed_current  – indices in the *current* (shifted) system
            3. before each batch, maps original→current by subtracting how many
            earlier removed orig indices were smaller than the candidate.
        """
        # --- build once ---
        full_ate_model   = self._create_full_dataset_model()
        original_full_ate = full_ate_model.get_original_ate()

        if len(knn_indices) == 0:
            if verbose:
                self._log("No neighbors found to remove")
            return original_full_ate, original_full_ate, []

        # --- state trackers ---
        removed_orig    = []   # indices in the original coordinate system
        removed_current = []   # indices in the shifting coordinate system
        current_ate     = original_full_ate

        if verbose:
            self._log(f"Original ATE (full dataset): {original_full_ate:.6f}")
            if target_ate is not None:
                self._log(f"Target ATE range: {target_ate-epsilon:.6f} – {target_ate+epsilon:.6f}")
            self._log(f"Found {len(knn_indices)} total neighbors to potentially remove")
            self._log(f"Removing in batches of {self.batch_size} points")

        # --- main loop ---
        sorted_candidates = sorted(knn_indices)
        for start in range(0, len(sorted_candidates), self.batch_size):
            prev_ate = current_ate
            batch_orig = sorted_candidates[start:start + self.batch_size]

            removed_orig.extend(batch_orig)

            if self.model_params["model_type"] == "linear":
                current_ate = full_ate_model.calculate_updated_ATE(
                    batch_orig, approx=approx
                )
            else:
                current_ate = full_ate_model.calculate_updated_ate(
                    batch_orig, method="retrain"
                )

            if verbose:
                self._log(
                    f"Batch {start//self.batch_size + 1}: "
                    f"Removed {len(batch_orig)} points, "
                    f"Total removed: {len(removed_orig)}, "
                    f"Current ATE: {current_ate:.6f}"
                )

            if (
                target_ate is not None
                and (target_ate - epsilon <= current_ate <= target_ate + epsilon or (current_ate - target_ate) * (prev_ate - target_ate) < 0)
            ):
                if verbose:
                    self._log("Target ATE range reached – stopping removal.")
                break

        return original_full_ate, current_ate, removed_orig
