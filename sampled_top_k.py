import pandas as pd
import numpy as np
import os
import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clustered_top_k import ClusteredATEBaselines
from ATE_update import ATEUpdateLinear, ATEUpdateLogistic

class SampledATEBaselines(ClusteredATEBaselines):
    """
    Scalable implementation of Top-k and Top-k+ baseline methods for ATE modification.
    
    This class implements a more efficient approach using cluster-based sampling:
    1. Cluster the full dataset into representative groups using K-means on (X, T, Y)
    2. Sample proportionally from each cluster to create a representative subset
    3. Run clustered top-k/top-k+ on the representative subset
    4. For each removed point from the subset, find k-nearest neighbors in the full dataset
    5. Remove these neighbors from the full dataset and compute final ATE
    
    This approach significantly reduces computational cost while maintaining better 
    representativeness compared to random sampling.
    
    New features:
    - Save sampled subset to CSV file for reuse
    - Load previously sampled subset from CSV file
    - Automatic caching and validation of sample files
    - **NEW**: Log verbose output to text files for analysis
    """

    def __init__(self, X, T, Y, model_type='linear', lambda_reg=0.1, max_iter=1000, find_confounders=False, sample_ratio=0.1, random_seed=42, sample_file_path='acs_sample.csv', use_cached_sample=True, save_sample=True, force_resample=False, log_dir='logs'):
        """
        Initialize the SampledATEBaselines with cluster-based sampling, caching, and logging.
        
        Args:
            X, T, Y: Full dataset features, treatment, and outcome
            model_type: 'linear' or 'logistic'
            lambda_reg: Regularization parameter for logistic regression
            max_iter: Maximum iterations for logistic regression
            find_confounders: Whether to find confounders automatically
            sample_ratio: Ratio of data to sample using cluster-based method (default 0.1 for 10%)
            random_seed: Random seed for reproducible clustering and sampling
            sample_file_path: Path to save/load the sample CSV file (default 'acs_sample.csv')
            use_cached_sample: Whether to try loading existing sample file first (default True)
            save_sample: Whether to save the sample to CSV after creating it (default True)
            force_resample: If True, ignore existing sample file and create new sample (default False)
            log_dir: Directory to save log files (default 'logs')
        """
        # Store original full dataset
        self.X_full = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T_full = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y_full = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        
        # Set sampling parameters
        self.sample_ratio = sample_ratio
        self.random_seed = random_seed
        self.sample_file_path = sample_file_path
        self.use_cached_sample = use_cached_sample
        self.save_sample = save_sample
        self.force_resample = force_resample
        
        # Setup logging directory
        self.log_dir = log_dir
        self.current_log_file = None
        self.log_file_handle = None
        self._ensure_log_dir()
        
        # Store model parameters for later use with full dataset
        self.model_params = {
            'model_type': model_type,
            'lambda_reg': lambda_reg,
            'max_iter': max_iter,
            'find_confounders': find_confounders
        }
        
        # Create or load cluster-based sampled subset
        self._handle_sampling()
        
        # Initialize parent class with sampled data
        super().__init__(self.X_sample, self.T_sample, self.Y_sample, 
                        model_type, lambda_reg, max_iter, find_confounders)
    
    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _setup_logging(self, log_file):
        """Setup logging to file."""
        self.current_log_file = os.path.join(self.log_dir, log_file)
        try:
            self.log_file_handle = open(self.current_log_file, 'w', encoding='utf-8')
            self._log(f"=== Log Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        except Exception as e:
            print(f"Warning: Could not setup logging to {self.current_log_file}: {e}")
            self.log_file_handle = None
    
    def _log(self, message):
        """Write message to log file and optionally print to console."""
        if self.log_file_handle:
            try:
                self.log_file_handle.write(message + '\n')
                self.log_file_handle.flush()  # Ensure immediate write
                # Also print to console for immediate feedback
                print(message)
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")
                print(message)  # Fallback to console only
        else:
            print(message)  # Fallback to console only
    
    def _close_logging(self):
        """Close the log file."""
        if self.log_file_handle:
            try:
                self._log(f"=== Log Ended at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                self.log_file_handle.close()
                print(f"Log saved to: {self.current_log_file}")
            except Exception as e:
                print(f"Warning: Could not close log file: {e}")
            finally:
                self.log_file_handle = None
    
    def _handle_sampling(self):
        """Handle the sampling process: try to load existing sample or create new one."""
        sample_loaded = False
        
        # Try to load existing sample if requested and not forcing resample
        if self.use_cached_sample and not self.force_resample:
            sample_loaded = self._try_load_cached_sample()
        
        # Create new sample if not loaded
        if not sample_loaded:
            print("Creating new cluster-based sample...")
            self._create_cluster_based_sample()
            
            # Save sample if requested
            if self.save_sample:
                self._save_sample_to_csv()
    
    def _try_load_cached_sample(self):
        """
        Try to load a previously saved sample from CSV file and metadata from JSON file.
        
        Returns:
            bool: True if sample was successfully loaded, False otherwise
        """
        if not os.path.exists(self.sample_file_path):
            print(f"Sample file '{self.sample_file_path}' not found. Will create new sample.")
            return False
        
        metadata_file_path = self._get_metadata_file_path()
        if not os.path.exists(metadata_file_path):
            print(f"Metadata file '{metadata_file_path}' not found. Will create new sample.")
            return False
        
        try:
            print(f"Loading cached sample from '{self.sample_file_path}'...")
            print(f"Loading metadata from '{metadata_file_path}'...")
            
            # Load the sample data
            sample_df = pd.read_csv(self.sample_file_path)
            
            # Load metadata from JSON file
            import json
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate the sample structure
            if not self._validate_sample_file(sample_df, metadata):
                print("Sample file validation failed. Will create new sample.")
                return False
            
            # Extract sample indices and data
            sample_indices = sample_df['sample_index'].values
            
            # Create sampled subset using the loaded indices
            self.sample_indices = np.array(sample_indices)
            self.X_sample = self.X_full.iloc[self.sample_indices].reset_index(drop=True)
            self.T_sample = self.T_full.iloc[self.sample_indices].reset_index(drop=True)
            self.Y_sample = self.Y_full.iloc[self.sample_indices].reset_index(drop=True)
            
            # Load metadata
            self.n_clusters_used = metadata.get('n_clusters_used', None)
            self.cluster_sample_info = metadata.get('cluster_sample_info', None)
            self.cluster_sample_targets = metadata.get('cluster_sample_targets', None)
            
            print(f"Successfully loaded cached sample: {len(self.sample_indices)} points")
            print(f"Sample ratio: {len(self.sample_indices)/len(self.X_full)*100:.1f}%")
            print(f"Metadata loaded: {len(metadata)} fields")
            
            return True
            
        except Exception as e:
            print(f"Error loading cached sample: {e}")
            print("Will create new sample.")
            return False
    
    def _validate_sample_file(self, sample_df, metadata=None):
        """
        Validate that the loaded sample file is compatible with current dataset.
        
        Args:
            sample_df: DataFrame loaded from sample CSV file
            metadata: Dictionary loaded from metadata JSON file
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['sample_index']
        
        # Check required columns in CSV
        for col in required_columns:
            if col not in sample_df.columns:
                print(f"Missing required column '{col}' in sample file.")
                return False
        
        # Check if sample indices are valid for current dataset
        max_index = sample_df['sample_index'].max()
        if max_index >= len(self.X_full):
            print(f"Sample file contains invalid indices (max: {max_index}, dataset size: {len(self.X_full)}).")
            return False
        
        # Check metadata if available
        if metadata is not None:
            # Check if sample ratio is reasonable
            sample_size = len(sample_df)
            current_ratio = sample_size / len(self.X_full)
            expected_ratio = metadata.get('sample_ratio', self.sample_ratio)
            
            if abs(current_ratio - expected_ratio) > 0.05:  # Allow 5% tolerance
                print(f"Sample ratio mismatch: file has {current_ratio:.3f}, expected {expected_ratio:.3f}")
                response = input("Continue with cached sample? (y/n): ").lower().strip()
                if response != 'y':
                    return False
            
            # Check dataset size consistency
            expected_full_size = metadata.get('full_dataset_size', len(self.X_full))
            if expected_full_size != len(self.X_full):
                print(f"Dataset size mismatch: metadata shows {expected_full_size}, current dataset has {len(self.X_full)}")
                response = input("Continue with cached sample? (y/n): ").lower().strip()
                if response != 'y':
                    return False
        
        return True
    
    def _save_sample_to_csv(self):
        """Save the current sample to CSV file and metadata to separate JSON file."""
        try:
            print(f"Saving sample to '{self.sample_file_path}'...")
            
            # Prepare sample data (without metadata)
            sample_data = {
                'sample_index': self.sample_indices
            }
            
            # Add feature columns from the sample
            for i, col in enumerate(self.X_sample.columns):
                sample_data[f'X_{col}'] = self.X_sample[col].values
            
            sample_data['T'] = self.T_sample.values
            sample_data['Y'] = self.Y_sample.values
            
            # Create DataFrame and save CSV
            sample_df = pd.DataFrame(sample_data)
            sample_df.to_csv(self.sample_file_path, index=False)
            
            # Save metadata to separate JSON file
            self._save_metadata_to_json()
            
            print(f"Sample saved successfully: {len(self.sample_indices)} points")
            print(f"CSV file: {self.sample_file_path}")
            print(f"Metadata file: {self._get_metadata_file_path()}")
            
        except Exception as e:
            print(f"Warning: Failed to save sample to CSV: {e}")
    
    def _save_metadata_to_json(self):
        """Save metadata to a separate JSON file."""
        # Helper function to convert numpy types to Python native types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types for JSON serialization."""
            if obj is None:
                return None
            elif isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float16, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (int, float, bool, str)):
                return obj
            else:
                # For any other type, try to convert to string as fallback
                try:
                    return str(obj)
                except:
                    return None
        
        # Prepare metadata
        metadata = {
            'sample_ratio': float(self.sample_ratio),
            'random_seed': int(self.random_seed),
            'full_dataset_size': int(len(self.X_full)),
            'sample_size': int(len(self.sample_indices)),
            'n_clusters_used': convert_numpy_types(getattr(self, 'n_clusters_used', None)),
            'cluster_sample_info': convert_numpy_types(getattr(self, 'cluster_sample_info', None)),
            'cluster_sample_targets': convert_numpy_types(getattr(self, 'cluster_sample_targets', None)) if hasattr(self, 'cluster_sample_targets') else None,
            'feature_names': list(self.X_full.columns),
            'created_timestamp': pd.Timestamp.now().isoformat(),
            'sampling_method': 'cluster-based'
        }
        
        # Convert the entire metadata to ensure all numpy types are handled
        metadata = convert_numpy_types(metadata)
        
        # Save to JSON file
        metadata_file_path = self._get_metadata_file_path()
        import json
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _get_metadata_file_path(self):
        """Get the metadata file path based on the sample file path."""
        # Replace .csv with _metadata.json
        if self.sample_file_path.endswith('.csv'):
            return self.sample_file_path.replace('.csv', '_metadata.json')
        else:
            return self.sample_file_path + '_metadata.json'
    
    def _create_cluster_based_sample(self):
        """Create a cluster-based representative subset of the full dataset."""
        n_samples = len(self.X_full)
        target_sample_size = int(n_samples * self.sample_ratio)
        
        # Ensure minimum sample size
        target_sample_size = max(target_sample_size, 10)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        print(f"Creating cluster-based sample: target size {target_sample_size} from {n_samples} points")
        
        # Prepare features for clustering (include treatment and outcome for better representation)
        scaler = StandardScaler()
        
        # Scale features
        X_scaled = scaler.fit_transform(self.X_full)
        
        # Scale treatment and outcome
        T_scaled = scaler.fit_transform(self.T_full.values.reshape(-1, 1))
        Y_scaled = scaler.fit_transform(self.Y_full.values.reshape(-1, 1))
        
        # Combine all attributes for clustering with appropriate weights
        # Give more weight to treatment and outcome as they are critical for ATE
        clustering_features = np.hstack([
            X_scaled,           # Features with weight 1
            T_scaled,           # Treatment with weight 2
            Y_scaled            # Outcome with weight 2
        ])
        
        # Determine optimal number of clusters
        # Rule: Use enough clusters to capture diversity but not too many to avoid empty clusters
        min_clusters = max(5, target_sample_size // 10)  # At least 5, but scale with sample size
        max_clusters = min(n_samples // 5, target_sample_size * 2)  # Don't exceed reasonable limits
        n_clusters = min(max_clusters, max(min_clusters, int(np.sqrt(target_sample_size) * 3)))
        
        print(f"Using {n_clusters} clusters for sampling")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        
        try:
            cluster_labels = kmeans.fit_predict(clustering_features)
        except Exception as e:
            print(f"Clustering failed: {e}, falling back to random sampling")
            self._fallback_random_sample(target_sample_size)
            return
        
        # Calculate cluster sizes and determine sampling strategy
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        
        # Strategy 1: Proportional sampling from each cluster
        total_points = np.sum(cluster_counts)
        cluster_sample_targets = np.round(cluster_counts * (target_sample_size / total_points)).astype(int)
        
        # Ensure we don't sample more than available in each cluster
        cluster_sample_targets = np.minimum(cluster_sample_targets, cluster_counts)
        
        # Adjust totals to match target
        current_total = np.sum(cluster_sample_targets)
        if current_total < target_sample_size:
            # Add samples to largest clusters first
            deficit = target_sample_size - current_total
            largest_clusters_idx = np.argsort(cluster_counts)[::-1]
            for i in range(deficit):
                cluster_idx = largest_clusters_idx[i % len(largest_clusters_idx)]
                if cluster_sample_targets[cluster_idx] < cluster_counts[cluster_idx]:
                    cluster_sample_targets[cluster_idx] += 1
        elif current_total > target_sample_size:
            # Remove samples from largest clusters first
            excess = current_total - target_sample_size
            largest_clusters_idx = np.argsort(cluster_counts)[::-1]
            for i in range(excess):
                cluster_idx = largest_clusters_idx[i % len(largest_clusters_idx)]
                if cluster_sample_targets[cluster_idx] > 0:
                    cluster_sample_targets[cluster_idx] -= 1
        
        # Sample from each cluster
        sampled_indices = []
        cluster_sample_info = {}
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            n_to_sample = cluster_sample_targets[i]
            
            cluster_sample_info[cluster_id] = {
                'total_size': len(cluster_indices),
                'sampled_size': n_to_sample
            }
            
            if n_to_sample > 0:
                if len(cluster_indices) <= n_to_sample:
                    # Take all points from small clusters
                    selected_indices = cluster_indices
                else:
                    # Randomly sample from larger clusters
                    selected_indices = np.random.choice(cluster_indices, size=n_to_sample, replace=False)
                
                sampled_indices.extend(selected_indices)
        
        # Store results
        self.sample_indices = np.array(sampled_indices)
        self.sample_indices = np.sort(self.sample_indices)  # Sort for consistency
        
        # Store clustering information
        self.n_clusters_used = n_clusters
        self.cluster_labels = cluster_labels
        self.cluster_sample_info = cluster_sample_info
        self.cluster_sample_targets = cluster_sample_targets
        
        print(f"Cluster-based sampling: selected {len(self.sample_indices)} points from {n_samples} total")
        print(f"Clusters used: {n_clusters}, Average points per cluster: {len(self.sample_indices)/n_clusters:.1f}")
        
        # Create sampled subset
        self.X_sample = self.X_full.iloc[self.sample_indices].reset_index(drop=True)
        self.T_sample = self.T_full.iloc[self.sample_indices].reset_index(drop=True)
        self.Y_sample = self.Y_full.iloc[self.sample_indices].reset_index(drop=True)
    
    def _fallback_random_sample(self, target_sample_size):
        """Fallback to random sampling if clustering fails."""
        print("Using random sampling as fallback")
        n_samples = len(self.X_full)
        target_sample_size = min(target_sample_size, n_samples)
        
        self.sample_indices = np.random.choice(n_samples, size=target_sample_size, replace=False)
        self.sample_indices = np.sort(self.sample_indices)
        
        self.X_sample = self.X_full.iloc[self.sample_indices].reset_index(drop=True)
        self.T_sample = self.T_full.iloc[self.sample_indices].reset_index(drop=True)
        self.Y_sample = self.Y_full.iloc[self.sample_indices].reset_index(drop=True)
        
        # Set fallback info
        self.n_clusters_used = None
        self.cluster_labels = None
        self.cluster_sample_info = None
        self.cluster_sample_targets = None
    
    def delete_cached_sample(self):
        """Delete the cached sample file and metadata file."""
        try:
            # Delete CSV file
            if os.path.exists(self.sample_file_path):
                os.remove(self.sample_file_path)
                print(f"Deleted cached sample file: {self.sample_file_path}")
            else:
                print(f"Sample file '{self.sample_file_path}' does not exist.")
            
            # Delete metadata file
            metadata_file_path = self._get_metadata_file_path()
            if os.path.exists(metadata_file_path):
                os.remove(metadata_file_path)
                print(f"Deleted metadata file: {metadata_file_path}")
            else:
                print(f"Metadata file '{metadata_file_path}' does not exist.")
                
        except Exception as e:
            print(f"Error deleting sample files: {e}")
    
    def get_sample_file_info(self):
        """Get information about the sample file and metadata file."""
        csv_exists = os.path.exists(self.sample_file_path)
        metadata_file_path = self._get_metadata_file_path()
        metadata_exists = os.path.exists(metadata_file_path)
        
        info = {
            "sample_file": {
                "exists": csv_exists,
                "path": self.sample_file_path
            },
            "metadata_file": {
                "exists": metadata_exists, 
                "path": metadata_file_path
            }
        }
        
        if not csv_exists and not metadata_exists:
            return info
        
        try:
            # Get CSV file info
            if csv_exists:
                file_stats = os.stat(self.sample_file_path)
                sample_df = pd.read_csv(self.sample_file_path)
                
                info["sample_file"].update({
                    "file_size_mb": file_stats.st_size / (1024 * 1024),
                    "created_time": pd.Timestamp.fromtimestamp(file_stats.st_birthtime),
                    "modified_time": pd.Timestamp.fromtimestamp(file_stats.st_mtime),
                    "sample_size": len(sample_df),
                    "columns": list(sample_df.columns)
                })
            
            # Get metadata file info
            if metadata_exists:
                file_stats = os.stat(metadata_file_path)
                import json
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                info["metadata_file"].update({
                    "file_size_mb": file_stats.st_size / (1024 * 1024),
                    "created_time": pd.Timestamp.fromtimestamp(file_stats.st_birthtime),
                    "modified_time": pd.Timestamp.fromtimestamp(file_stats.st_mtime),
                    "metadata": metadata
                })
            
            return info
            
        except Exception as e:
            info["error"] = f"Could not read files: {e}"
            return info
    
    def _find_knn_in_full_dataset(self, removed_sample_indices, k_neighbors=5, verbose=False):
        """
        For each removed point from the sampled subset, find k-nearest neighbors in the full dataset.
        Uses the same feature combination (X, T, Y) as the clustering step for consistency.
        
        Args:
            removed_sample_indices: Indices of removed points in the sampled subset
            k_neighbors: Number of nearest neighbors to find for each removed point
            verbose: Whether to log progress information
            
        Returns:
            List of unique indices in the full dataset to be removed
        """
        if not removed_sample_indices:
            return []
        
        # Convert sample indices to original dataset indices
        original_removed_indices = [self.sample_indices[i] for i in removed_sample_indices]
        
        if verbose:
            self._log(f"Finding KNN for {len(original_removed_indices)} removed points...")
        
        # Prepare features for KNN - use same combination as clustering (X, T, Y)
        scaler = StandardScaler()
        
        # Scale all features
        X_full_scaled = scaler.fit_transform(self.X_full)
        T_full_scaled = scaler.fit_transform(self.T_full.values.reshape(-1, 1))
        Y_full_scaled = scaler.fit_transform(self.Y_full.values.reshape(-1, 1))
        
        # Combine all attributes with same weights as clustering
        full_features = np.hstack([
            X_full_scaled,
            T_full_scaled,
            Y_full_scaled
        ])
        
        # Get combined features of removed points
        removed_points_features = full_features[original_removed_indices]
        
        # Create KNN model - use k_neighbors+1 to account for the point itself potentially being included
        actual_k = min(k_neighbors + len(original_removed_indices), len(self.X_full))
        knn = NearestNeighbors(n_neighbors=actual_k, metric='euclidean', n_jobs=-1)
        knn.fit(full_features)
        
        # Find neighbors for each removed point
        all_neighbors = set()
        
        for i, removed_point in enumerate(removed_points_features):
            distances, indices = knn.kneighbors([removed_point])
            
            # Filter out the removed points themselves and select top k_neighbors
            valid_neighbors = []
            for idx in indices[0]:
                if idx not in original_removed_indices and len(valid_neighbors) < k_neighbors:
                    valid_neighbors.append(idx)
            
            all_neighbors.update(valid_neighbors)
            
            if verbose and (i + 1) % 100 == 0:
                self._log(f"Processed {i + 1}/{len(removed_points_features)} points")
        
        result_neighbors = list(all_neighbors)
        
        if verbose:
            self._log(f"Found {len(result_neighbors)} unique neighbors from {len(original_removed_indices)} removed points")
            self._log(f"KNN search used combined features: X + T*2 + Y*2 (same as clustering)")
        
        return result_neighbors
    
    def _create_full_dataset_model(self):
        """Create an ATE model for the full dataset."""
        if self.model_params['model_type'].lower() == 'linear':
            return ATEUpdateLinear(self.X_full, self.T_full, self.Y_full, 
                                 find_confounders=self.model_params['find_confounders'])
        elif self.model_params['model_type'].lower() == 'logistic':
            return ATEUpdateLogistic(self.X_full, self.T_full, self.Y_full,
                                   C=self.model_params['lambda_reg'],
                                   max_iter=self.model_params['max_iter'],
                                   find_confounders=self.model_params['find_confounders'])
    
    def sampled_top_k(self, target_ate=None, epsilon=0.01, k=None, k_neighbors=5,
                      approx=False, use_clustering=True, n_clusters=None, 
                      samples_per_cluster=2, verbose=True, log_file=None):
        """
        Scalable Top-k method using cluster-based sampling and KNN.
        
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
                log_file = f"sampled_top_k_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Sampled Top-k Algorithm (Cluster-based Sampling) ===")
            self._log(f"Sample ratio: {self.sample_ratio*100}%")
            self._log(f"Sample size: {len(self.X_sample)}")
            self._log(f"Full dataset size: {len(self.X_full)}")
            self._log(f"Clusters used for sampling: {getattr(self, 'n_clusters_used', 'N/A')}")
            self._log(f"K-neighbors: {k_neighbors}")
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
        
        # Step 3: Calculate final ATE on full dataset
        if verbose:
            self._log(f"\n--- Step 3: Calculating final ATE on full dataset ---")
        
        full_ate_model = self._create_full_dataset_model()
        original_full_ate = full_ate_model.get_original_ate()
        
        if len(knn_indices) > 0:
            if self.model_params['model_type'] == 'linear':
                final_ate = full_ate_model.calculate_updated_ATE(knn_indices, approx=approx)
            else:
                final_ate = full_ate_model.calculate_updated_ate(knn_indices, method='retrain')
        else:
            final_ate = original_full_ate
        
        # Compile results
        result = {
            'removed_indices': knn_indices,
            'final_ate': final_ate,
            'num_removed': len(knn_indices),
            'original_ate': original_full_ate,
            'ate_difference': final_ate - original_full_ate,
            'sample_result': sample_result,
            'sample_size': len(self.X_sample),
            'full_dataset_size': len(self.X_full),
            'k_neighbors': k_neighbors,
            'sample_ratio': self.sample_ratio,
            'computational_efficiency': len(self.X_sample) / len(self.X_full),
            'sampling_method': 'cluster-based',
            'log_file': self.current_log_file if verbose else None
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE (full dataset): {original_full_ate:.6f}")
            self._log(f"Final ATE: {final_ate:.6f}")
            self._log(f"ATE Difference: {final_ate - original_full_ate:.6f}")
            self._log(f"Points removed from full dataset: {len(knn_indices)}")
            self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result
    
    def sampled_top_k_plus(self, target_ate=None, epsilon=0.01, k=None, k_neighbors=5,
                          approx=False, use_clustering=True, n_clusters=None,
                          samples_per_cluster=2, verbose=True, log_file=None):
        """
        Scalable Top-k+ method using cluster-based sampling and KNN.
        
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
                log_file = f"sampled_top_k_plus_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Sampled Top-k+ Algorithm (Cluster-based Sampling) ===")
            self._log(f"Sample ratio: {self.sample_ratio*100}%")
            self._log(f"Sample size: {len(self.X_sample)}")
            self._log(f"Full dataset size: {len(self.X_full)}")
            self._log(f"Clusters used for sampling: {getattr(self, 'n_clusters_used', 'N/A')}")
            self._log(f"K-neighbors: {k_neighbors}")
            self._log(f"Log file: {self.current_log_file}")
        
        # Step 1: Run clustered top-k+ on cluster-based sampled subset
        if verbose:
            self._log(f"\n--- Step 1: Running clustered top-k+ on cluster-based sampled subset ---")
        
        sample_result = super().top_k_plus(target_ate=target_ate, epsilon=epsilon, k=k,
                                          approx=approx, use_clustering=use_clustering,
                                          n_clusters=n_clusters, samples_per_cluster=samples_per_cluster,
                                          verbose=verbose)
        
        # Step 2: Find k-nearest neighbors in full dataset
        if verbose:
            self._log(f"\n--- Step 2: Finding nearest neighbors in full dataset ---")
        
        knn_indices = self._find_knn_in_full_dataset(
            sample_result['removed_indices'], k_neighbors, verbose=verbose)
        
        if verbose:
            self._log(f"Found {len(knn_indices)} unique nearest neighbors to remove")
        
        # Step 3: Calculate final ATE on full dataset
        if verbose:
            self._log(f"\n--- Step 3: Calculating final ATE on full dataset ---")
        
        full_ate_model = self._create_full_dataset_model()
        original_full_ate = full_ate_model.get_original_ate()
        
        if len(knn_indices) > 0:
            if self.model_params['model_type'] == 'linear':
                final_ate = full_ate_model.calculate_updated_ATE(knn_indices, approx=approx)
            else:
                final_ate = full_ate_model.calculate_updated_ate(knn_indices, method='retrain')
        else:
            final_ate = original_full_ate
        
        # Compile results
        result = {
            'removed_indices': knn_indices,
            'final_ate': final_ate,
            'num_removed': len(knn_indices),
            'original_ate': original_full_ate,
            'ate_difference': final_ate - original_full_ate,
            'sample_result': sample_result,
            'sample_size': len(self.X_sample),
            'full_dataset_size': len(self.X_full),
            'k_neighbors': k_neighbors,
            'sample_ratio': self.sample_ratio,
            'computational_efficiency': len(self.X_sample) / len(self.X_full),
            'sampling_method': 'cluster-based',
            'log_file': self.current_log_file if verbose else None
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE (full dataset): {original_full_ate:.6f}")
            self._log(f"Final ATE: {final_ate:.6f}")
            self._log(f"ATE Difference: {final_ate - original_full_ate:.6f}")
            self._log(f"Points removed from full dataset: {len(knn_indices)}")
            self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result
    
    def get_sample_info(self):
        """Get information about the cluster-based sampling process."""
        info = {
            'sample_ratio': self.sample_ratio,
            'sample_size': len(self.X_sample),
            'full_dataset_size': len(self.X_full),
            'sample_indices': self.sample_indices,
            'random_seed': self.random_seed,
            'computational_efficiency': self.sample_ratio,
            'sampling_method': 'cluster-based',
            'n_clusters_used': getattr(self, 'n_clusters_used', None),
            'sample_file_path': self.sample_file_path,
            'cached_sample_used': hasattr(self, '_sample_loaded_from_cache'),
            'log_dir': self.log_dir
        }
        
        # Add cluster-specific information if available
        if hasattr(self, 'cluster_sample_info') and self.cluster_sample_info is not None:
            cluster_sizes = [info['total_size'] for info in self.cluster_sample_info.values()]
            sampled_sizes = [info['sampled_size'] for info in self.cluster_sample_info.values()]
            
            info['cluster_distribution'] = {
                'total_clusters': len(cluster_sizes),
                'min_cluster_size': np.min(cluster_sizes),
                'max_cluster_size': np.max(cluster_sizes),
                'avg_cluster_size': np.mean(cluster_sizes),
                'std_cluster_size': np.std(cluster_sizes),
                'min_sampled_size': np.min(sampled_sizes),
                'max_sampled_size': np.max(sampled_sizes),
                'avg_sampled_size': np.mean(sampled_sizes),
                'std_sampled_size': np.std(sampled_sizes)
            }
        
        return info
    
    def get_cluster_analysis(self):
        """Get detailed cluster analysis information."""
        if not hasattr(self, 'cluster_sample_info') or self.cluster_sample_info is None:
            return "No cluster information available (fallback sampling was used)"
        
        analysis = {
            'n_clusters': self.n_clusters_used,
            'total_sampled': len(self.sample_indices),
            'cluster_details': self.cluster_sample_info
        }
        
        return analysis
