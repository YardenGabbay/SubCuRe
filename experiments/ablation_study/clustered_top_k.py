import pandas as pd
import numpy as np
import os
import sys
import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import datetime
from dowhy import CausalModel


class ClusteredATEBaselines:
    """
    Implementation of Top-k and Top-k+ baseline methods for ATE modification.
    These methods identify and remove data points to achieve a target ATE range.
    Modified to consider the direction of influence toward the target ATE.
    Enhanced with clustering-based sampling to reduce computational cost.
    **NEW**: Log verbose output to text files for analysis
    """
    
    def __init__(self, X, T, Y,treatment,outcome,confounders,df, model_type='linear', lambda_reg=0.1, max_iter=1000, find_confounders=False, log_dir='logs'):
        # Store the dataset
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        self.df=df
        self.treatment=treatment
        self.outcome=outcome
        self.confounders=confounders

        # Setup logging directory
        self.log_dir = log_dir
        self.current_log_file = None
        self.log_file_handle = None
        self._ensure_log_dir()
        
        # Initialize the appropriate ATE update model
        # Store the original ATE
        self.original_ate = self.calc_ate(treatment,outcome,confounders,df)
        self.model_type = model_type
    
    def calc_ate(self, treatment, outcome, confounders, data):
        model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
        )

        identified_estimand = model.identify_effect()

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        return estimate.value
    
    def get_ate_diff(self, treatment, outcome, confounders, data, df_ate):
        model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
        )

        identified_estimand = model.identify_effect()

        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        return estimate.value - df_ate

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
    
    def _create_clusters_and_sample(self, n_clusters=None, samples_per_cluster=2, verbose=False):
        """
        Create clusters based on confounders, treatment, and outcome values,
        then sample representatives from each cluster.
        
        Args:
            n_clusters: Number of clusters (if None, automatically determined)
            samples_per_cluster: Number of samples to take from each cluster
            verbose: Whether to print clustering information
        
        Returns:
            dict: Mapping from original indices to their cluster representatives
        """
        n_samples = len(self.X)
        
        # Combine features, treatment, and outcome for clustering
        # Normalize treatment and outcome to be on similar scale as features
        scaler = StandardScaler()
        
        # Prepare the feature matrix for clustering
        X_scaled = scaler.fit_transform(self.X)
        T_scaled = scaler.fit_transform(self.T.values.reshape(-1, 1))
        Y_scaled = scaler.fit_transform(self.Y.values.reshape(-1, 1))
        
        # Combine all attributes
        clustering_features = np.hstack([X_scaled, T_scaled, Y_scaled])
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            # Use heuristic: sqrt(n_samples) with min 5 and max n_samples//10
            n_clusters = max(5, min(int(np.sqrt(n_samples)), n_samples // 10))

        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, n_samples)
        
        if verbose:
            self._log(f"Creating {n_clusters} clusters from {n_samples} data points")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_features)
        
        # Create cluster mapping and sample representatives
        cluster_mapping = {}  # maps original_idx -> representative_idx
        cluster_representatives = {}  # maps cluster_id -> list of representative indices
        
        for cluster_id in range(n_clusters):
            # Get all indices in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Sample representatives from this cluster
            if len(cluster_indices) <= samples_per_cluster:
                # If cluster is small, take all points as representatives
                representatives = cluster_indices.tolist()
            else:
                # Sample diverse representatives based on distance to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_data = clustering_features[cluster_indices]
                
                # Calculate distances to cluster center
                distances = euclidean_distances([cluster_center], cluster_data)[0]
                
                # Select representatives: closest to center + some diverse ones
                closest_idx = np.argmin(distances)
                representatives = [cluster_indices[closest_idx]]
                
                # Add additional diverse representatives if needed
                remaining_samples = samples_per_cluster - 1
                if remaining_samples > 0 and len(cluster_indices) > 1:
                    # Select points with varying distances from center
                    distance_percentiles = np.linspace(25, 75, remaining_samples)
                    for percentile in distance_percentiles:
                        target_distance = np.percentile(distances, percentile)
                        diverse_idx = np.argmin(np.abs(distances - target_distance))
                        if cluster_indices[diverse_idx] not in representatives:
                            representatives.append(cluster_indices[diverse_idx])
            
            cluster_representatives[cluster_id] = representatives
            
            # Map all cluster members to their representatives
            for idx in cluster_indices:
                # Assign each point to the closest representative in its cluster
                if len(representatives) == 1:
                    cluster_mapping[idx] = representatives[0]
                else:
                    # Find closest representative
                    point_data = clustering_features[idx:idx+1]
                    rep_distances = euclidean_distances(point_data, 
                                                      clustering_features[representatives])[0]
                    closest_rep_idx = np.argmin(rep_distances)
                    cluster_mapping[idx] = representatives[closest_rep_idx]
        
        if verbose:
            total_representatives = sum(len(reps) for reps in cluster_representatives.values())
            self._log(f"Selected {total_representatives} representatives from {n_clusters} clusters")
            self._log(f"Computational reduction: {total_representatives}/{n_samples} = {total_representatives/n_samples:.3f}")
        
        return cluster_mapping, cluster_representatives
    
    def top_k(self, target_ate=None, epsilon=0.01, k=None, approx=False, 
              use_clustering=True, n_clusters=None, samples_per_cluster=2, verbose=True, log_file=None):
        """
        Top-k method with optional clustering-based sampling for efficiency.
        
        Args:
            target_ate: Target ATE value
            epsilon: Tolerance for target ATE
            k: Maximum number of points to remove (if target_ate is None)
            approx: Whether to use approximation for ATE calculation
            use_clustering: Whether to use clustering-based sampling
            n_clusters: Number of clusters (if None, automatically determined)
            samples_per_cluster: Number of samples per cluster
            verbose: Whether to log progress information
            log_file: Custom log file name (if None, auto-generated)
            
        Returns:
            Dictionary containing results of the algorithm
        """
        if target_ate is None and k is None:
            raise ValueError("Either target_ate or k must be provided")
        
        # Setup logging
        if verbose:
            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"clustered_top_k_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Clustered Top-k Algorithm ===")
            self._log(f"Dataset size: {len(self.X)}")
            self._log(f"Use clustering: {use_clustering}")
            if use_clustering:
                self._log(f"Samples per cluster: {samples_per_cluster}")
            self._log(f"Model type: {self.model_type}")
            self._log(f"Log file: {self.current_log_file}")
        
        if verbose:
            self._log(f"Original ATE: {self.original_ate}")
            if target_ate:
                self._log(f"Target ATE range: {target_ate - epsilon} to {target_ate + epsilon}")
        
        n_samples = len(self.X)
        
        # Use clustering-based sampling if enabled
        if use_clustering:
            if verbose:
                self._log(f"\n--- Step 1: Creating clusters and sampling representatives ---")
            
            cluster_mapping, cluster_representatives = self._create_clusters_and_sample(
                n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, verbose=verbose)
            
            if verbose:
                self._log(f"\n--- Step 2: Calculating influence scores for representatives ---")
            
            # Calculate influence scores only for representatives
            representative_indices = set()
            for reps in cluster_representatives.values():
                representative_indices.update(reps)
            
            representative_scores = {}
            for i, rep_idx in enumerate(representative_indices):
                remain_df = self.df.drop([rep_idx])
                ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                representative_scores[rep_idx] = ate_diff
                
                if verbose and (i + 1) % 100 == 0:
                    self._log(f"Calculated influence for {i + 1}/{len(representative_indices)} representatives")
            
            if verbose:
                self._log(f"\n--- Step 3: Mapping influence scores to all data points ---")
            
            # Create influence scores for all points using their representatives
            influence_scores = []
            for i in range(n_samples):
                rep_idx = cluster_mapping[i]
                rep_score = representative_scores[rep_idx]
                influence_scores.append((i, rep_score))
        else:
            if verbose:
                self._log(f"\n--- Calculating influence scores for all data points ---")
            
            # Original method: calculate influence for all points
            influence_scores = []
            for i in range(n_samples):
                remain_df = self.df.drop([i])
                ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                influence_scores.append((i, ate_diff))
                
                if verbose and (i + 1) % 500 == 0:
                    self._log(f"Calculated influence for {i + 1}/{n_samples} points")
        
        # Rest of the algorithm remains the same
        initial_direction = 1 if (target_ate > self.original_ate) else -1
        
        if verbose:
            self._log(f"\n--- Step 4: Starting iterative removal process ---")
            if initial_direction == 1:
                self._log(f"Initial direction: Increasing ATE toward target")
            else:
                self._log(f"Initial direction: Decreasing ATE toward target") 
        
        # Create two sorted lists - one for each direction of influence
        increasing_ate = [(i, diff) for i, diff in influence_scores if diff > 0]
        decreasing_ate = [(i, diff) for i, diff in influence_scores if diff < 0]
        
        # Sort by magnitude of influence (largest first)
        increasing_ate.sort(key=lambda x: abs(x[1]), reverse=True)
        decreasing_ate.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if verbose:
            self._log(f"Points that increase ATE: {len(increasing_ate)}")
            self._log(f"Points that decrease ATE: {len(decreasing_ate)}")
        
        # Initialize tracking variables
        removed_indices = []
        current_ate = self.original_ate
        ate_history = [self.original_ate]
        num_removed_history = [0]
        used_indices = set()
        
        # Determine stopping condition
        if k is not None:
            max_removals = min(k, n_samples)
            stopping_condition = lambda ate: len(removed_indices) >= max_removals
            if verbose:
                self._log(f"Stopping condition: Remove up to {max_removals} points")
        else:
            stopping_condition = lambda ate: target_ate - epsilon <= ate <= target_ate + epsilon
            if verbose:
                self._log(f"Stopping condition: Reach target ATE within epsilon tolerance")
        
        # Run until stopping condition is met
        iteration = 0
        while not stopping_condition(current_ate) and len(removed_indices) < n_samples:
            iteration += 1
            current_direction = 1 if (target_ate > current_ate) else -1
            
            if current_direction != initial_direction and verbose:
                self._log(f"\nIteration {iteration}: Direction changed")
                self._log(f"Current ATE: {current_ate}, Target: {target_ate}")
                self._log(f"Switching from {'increasing' if initial_direction > 0 else 'decreasing'} to {'increasing' if current_direction > 0 else 'decreasing'} ATE")
            
            candidates = increasing_ate if current_direction > 0 else decreasing_ate
            
            next_point = None
            for idx, diff in candidates:
                if idx not in used_indices:
                    next_point = (idx, diff)
                    break
            
            if next_point is None:
                if verbose:
                    self._log(f"Iteration {iteration}: No unused points left that move ATE in desired direction")
                other_candidates = decreasing_ate if current_direction > 0 else increasing_ate
                for idx, diff in other_candidates:
                    if idx not in used_indices:
                        next_point = (idx, diff)
                        if verbose:
                            self._log(f"Using point that moves ATE in opposite direction (least harmful)")
                        break
            
            if next_point is None:
                if verbose:
                    self._log(f"Iteration {iteration}: No more unused points available. Stopping.")
                break
            
            idx, diff = next_point
            removed_indices.append(idx)
            used_indices.add(idx)
            
            self.df = self.df.drop([idx])
            current_ate = self.calc_ate(self.treatment, self.outcome,self.confounders, self.df)

            ate_history.append(current_ate)
            num_removed_history.append(len(removed_indices))
            
            if verbose and len(removed_indices) % 500 == 0:
                self._log(f"Iteration {iteration}: Removed {len(removed_indices)} points, Current ATE: {current_ate}")
            elif verbose and len(removed_indices) <= 10:
                self._log(f"Iteration {iteration}: Removed point {idx} (influence: {diff:.6f}), Current ATE: {current_ate}")
        
        result = {
            'removed_indices': removed_indices,
            'final_ate': current_ate,
            'num_removed': len(removed_indices),
            'original_ate': self.original_ate,
            'ate_difference': current_ate - self.original_ate,
            'ate_history': ate_history,
            'num_removed_history': num_removed_history,
            'use_clustering': use_clustering,
            'computational_efficiency': (len(representative_indices) / n_samples) if use_clustering else 1.0,
            'log_file': self.current_log_file if verbose else None
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE: {self.original_ate}")
            self._log(f"Final ATE: {current_ate}")
            self._log(f"ATE Difference: {current_ate - self.original_ate}")
            self._log(f"Removed {len(removed_indices)} data points")
            if use_clustering:
                self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result
    
    def top_k_plus(self, target_ate=None, epsilon=0.01, k=None, approx=False,
               use_clustering=True, n_clusters=None, samples_per_cluster=2, 
               influence_recalc_interval=1, verbose=True, log_file=None):
        """
        Top-k+ method with optional clustering-based sampling for efficiency.
        Clustering is performed once at the beginning, then in each iteration,
        cluster-level influence scores are recalculated and clusters are ranked.
        
        Args:
            target_ate: Target ATE value
            epsilon: Tolerance for target ATE
            k: Maximum number of points to remove (if target_ate is None)
            approx: Whether to use approximation for ATE calculation
            use_clustering: Whether to use clustering-based sampling
            n_clusters: Number of clusters (if None, automatically determined)
            samples_per_cluster: Number of samples per cluster
            influence_recalc_interval: How often to recalculate influence scores (1=every iteration, 
                                    100=every 100 iterations, 0=only once at start)
            verbose: Whether to log progress information
            log_file: Custom log file name (if None, auto-generated)
            
        Returns:
            Dictionary containing results of the algorithm
        """
        if target_ate is None and k is None:
            raise ValueError("Either target_ate or k must be provided")
        
        # Setup logging
        if verbose:
            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"clustered_top_k_plus_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Clustered Top-k+ Algorithm ===")
            self._log(f"Dataset size: {len(self.X)}")
            self._log(f"Use clustering: {use_clustering}")
            if use_clustering:
                self._log(f"Samples per cluster: {samples_per_cluster}")
            self._log(f"Influence recalc interval: {influence_recalc_interval}")
            self._log(f"Model type: {self.model_type}")
            self._log(f"Log file: {self.current_log_file}")
        
        if verbose:
            self._log(f"Original ATE: {self.original_ate}")
            if target_ate:
                self._log(f"Target ATE range: {target_ate - epsilon} to {target_ate + epsilon}")
        
        n_samples = len(self.X)
        
        # Pre-cluster the data once
        if use_clustering:
            if verbose:
                self._log(f"\n--- Step 1: Creating clusters ---")
            
            cluster_mapping, cluster_representatives = self._create_clusters_and_sample(
                n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, verbose=verbose)
            
            # Create cluster membership tracking (simplified)
            clusters = {}  # cluster_id -> set of original indices
            
            # Initialize empty clusters
            for cluster_id in cluster_representatives.keys():
                clusters[cluster_id] = set()
            
            # Assign points to clusters based on their representatives
            for original_idx, rep_idx in cluster_mapping.items():
                # Find which cluster this representative belongs to
                for cluster_id, representatives in cluster_representatives.items():
                    if rep_idx in representatives:
                        clusters[cluster_id].add(original_idx)
                        break
            
            if verbose:
                cluster_sizes = [len(members) for members in clusters.values()]
                self._log(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        
        # Initialize iteration variables
        removed_indices = []
        current_ate = self.original_ate
        available_indices = set(range(n_samples))
        ate_history = [self.original_ate]
        num_removed_history = [0]
        
        # Initialize influence score caching
        cached_influence_scores = {}
        last_recalc_iteration = 0
        
        # Set random seed for reproducible sampling
        np.random.seed(42)
        
        # Determine stopping condition
        if k is not None:
            max_removals = min(k, n_samples)
            stopping_condition = lambda ate: len(removed_indices) >= max_removals
            if verbose:
                self._log(f"Stopping condition: Remove up to {max_removals} points")
        else:
            stopping_condition = lambda ate: target_ate - epsilon <= ate <= target_ate + epsilon
            if verbose:
                self._log(f"Stopping condition: Reach target ATE within epsilon tolerance")
        
        overshot = False
        prev_desired_direction = 1 if (target_ate > self.original_ate) else -1
        iteration = 0
        
        if verbose:
            self._log(f"\n--- Step 2: Starting iterative removal process ---")
        
        while not stopping_condition(current_ate) and available_indices:
            iteration += 1
            desired_direction = 1 if (target_ate > current_ate) else -1
            
            if desired_direction != prev_desired_direction:
                overshot = True
                if verbose:
                    self._log(f"\nIteration {iteration}: Direction changed - Overshot the target")
                    self._log(f"Current ATE: {current_ate}, Target: {target_ate}")
                    self._log(f"Reversing direction from {prev_desired_direction} to {desired_direction} to move back toward target")
            
            prev_desired_direction = desired_direction
            
            if verbose and iteration == 1:
                if desired_direction == 1:
                    self._log(f"Initial direction: Increasing ATE toward target")
                else:
                    self._log(f"Initial direction: Decreasing ATE toward target")
            
            # Determine if we need to recalculate influence scores
            need_recalc = (iteration == 1 or  (influence_recalc_interval > 0 and (iteration - last_recalc_iteration) >= influence_recalc_interval))
            
            if use_clustering:
                if need_recalc:
                    if verbose:
                        self._log(f"\nIteration {iteration}: Recalculating cluster-level influence scores...")
                    
                    # Calculate cluster-level influence scores by sampling points from each cluster
                    cluster_scores = {}
                    available_clusters = set()
                    
                    for cluster_id in clusters:
                        # Find available points in this cluster
                        available_in_cluster = list(clusters[cluster_id].intersection(available_indices))
                        if not available_in_cluster:
                            continue  # Skip empty clusters
                        
                        available_clusters.add(cluster_id)

                        # Sample up to 20 points from this cluster to calculate influence
                        n_samples_cluster = min(20, len(available_in_cluster))
                        sampled_points = np.random.choice(available_in_cluster, size=n_samples_cluster, replace=False)
                        
                        cluster_influence_scores = []
                        for sample_idx in sampled_points:
                            remain_df = self.df.drop([sample_idx])
                            ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                            cluster_influence_scores.append(ate_diff)
                        
                        # Use average influence of sampled points as cluster score
                        cluster_scores[cluster_id] = np.mean(cluster_influence_scores)
                    
                    # Cache the scores
                    cached_influence_scores = cluster_scores
                    last_recalc_iteration = iteration
                    
                    if verbose:
                        self._log(f"Calculated influence scores for {len(cluster_scores)} clusters")
                else:
                    # Use cached scores, but filter for available clusters
                    cluster_scores = {}
                    for cluster_id in clusters:
                        available_in_cluster = list(clusters[cluster_id].intersection(available_indices))
                        if available_in_cluster and cluster_id in cached_influence_scores:
                            cluster_scores[cluster_id] = cached_influence_scores[cluster_id]
                    
                    if verbose and iteration % 100 == 1:
                        self._log(f"Iteration {iteration}: Using cached influence scores ({len(cluster_scores)} available clusters)")
                
                # Filter clusters that move ATE in the desired direction
                valid_clusters = [(cluster_id, score) for cluster_id, score in cluster_scores.items() if score * desired_direction > 0]
                
                if not valid_clusters:
                    if verbose:
                        self._log(f"Iteration {iteration}: No clusters move ATE in the desired direction, selecting least harmful cluster")
                    valid_clusters = sorted(cluster_scores.items(), key=lambda x: abs(x[1]))
                    if not valid_clusters:
                        if verbose:
                            self._log(f"Iteration {iteration}: No more clusters available. Stopping.")
                        break
                
                # Find the cluster with highest influence in the desired direction
                if valid_clusters[0][1] * desired_direction > 0:
                    best_cluster_id = max(valid_clusters, key=lambda x: abs(x[1]) if x[1] * desired_direction > 0 else 0)[0]
                else:
                    best_cluster_id = valid_clusters[0][0]
                
                # Randomly sample one point from the best cluster to remove
                available_in_best_cluster = list(clusters[best_cluster_id].intersection(available_indices))
                next_idx = np.random.choice(available_in_best_cluster)
                
                if verbose and iteration <= 10:
                    best_score = cluster_scores[best_cluster_id]
                    recalc_status = "RECALCULATED" if need_recalc else "CACHED"
                    self._log(f"Iteration {iteration}: Selected cluster {best_cluster_id} (score: {best_score:.6f} [{recalc_status}]), removing point {next_idx}")
            
            else:
                if need_recalc:
                    if verbose:
                        self._log(f"\nIteration {iteration}: Recalculating influence scores for available points...")
                    
                    # Original method: calculate influence for all available points
                    influence_scores = []
                    for i in available_indices:
                        remain_df = self.df.drop([i])
                        ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                        influence_scores.append((i, ate_diff))
                    
                    # Cache the scores
                    cached_influence_scores = {idx: diff for idx, diff in influence_scores}
                    last_recalc_iteration = iteration
                    
                    if verbose:
                        self._log(f"Calculated influence scores for {len(influence_scores)} points")
                else:
                    # Use cached scores, but filter for available points
                    influence_scores = [(idx, diff) for idx, diff in cached_influence_scores.items() if idx in available_indices]
                    
                    if verbose and iteration % 100 == 1:
                        self._log(f"Iteration {iteration}: Using cached influence scores ({len(influence_scores)} available points)")
                
                # Filter points that move ATE in the desired direction
                valid_influences = [(idx, diff) for idx, diff in influence_scores if diff * desired_direction > 0]
                
                if not valid_influences:
                    if verbose:
                        self._log(f"Iteration {iteration}: No points move ATE in the desired direction, selecting least harmful point")
                    valid_influences = sorted(influence_scores, key=lambda x: abs(x[1]))
                    if not valid_influences:
                        if verbose:
                            self._log(f"Iteration {iteration}: No more points available. Stopping.")
                        break
                
                # Find the point with highest influence in the desired direction
                if valid_influences[0][1] * desired_direction > 0:
                    next_idx = max(valid_influences, key=lambda x: abs(x[1]) if x[1] * desired_direction > 0 else 0)[0]
                    next_influence = next((diff for idx, diff in valid_influences if idx == next_idx), 0)
                else:
                    next_idx = valid_influences[0][0]
                    next_influence = valid_influences[0][1]
                
                if verbose and iteration <= 10:
                    recalc_status = "RECALCULATED" if need_recalc else "CACHED"
                    self._log(f"Iteration {iteration}: Selected point {next_idx} (influence: {next_influence:.6f} [{recalc_status}])")
            
            # Remove the selected point
            removed_indices.append(next_idx)
            available_indices.remove(next_idx)
            
            # Update the model and get new ATE
            self.df = self.df.drop([next_idx])
            current_ate = self.calc_ate(self.treatment, self.outcome,self.confounders, self.df)

            ate_history.append(current_ate)
            num_removed_history.append(len(removed_indices))
            
            if verbose and len(removed_indices) % 500 == 0:
                self._log(f"Iteration {iteration}: Removed {len(removed_indices)} points, Current ATE: {current_ate}")
            
            if stopping_condition(current_ate):
                if verbose:
                    self._log(f"Iteration {iteration}: Stopping condition met!")
                break
        
        result = {
            'removed_indices': removed_indices,
            'final_ate': current_ate,
            'num_removed': len(removed_indices),
            'original_ate': self.original_ate,
            'ate_difference': current_ate - self.original_ate,
            'ate_history': ate_history,
            'num_removed_history': num_removed_history,
            'use_clustering': use_clustering,
            'computational_efficiency': (len(cluster_representatives) if use_clustering else n_samples) / n_samples,
            'influence_recalc_interval': influence_recalc_interval,
            'total_influence_recalculations': (iteration - 1) // max(1, influence_recalc_interval) + 1 if influence_recalc_interval > 0 else 1,
            'log_file': self.current_log_file if verbose else None
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE: {self.original_ate}")
            self._log(f"Final ATE: {current_ate}")
            self._log(f"ATE Difference: {current_ate - self.original_ate}")
            self._log(f"Removed {len(removed_indices)} data points")
            self._log(f"Influence recalculations: {result['total_influence_recalculations']} times")
            if use_clustering:
                self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result

    def top_k_plus_best(self, target_ate=None, epsilon=0.01, k=None, approx=False, use_clustering=True, n_clusters=None, samples_per_cluster=2,  influence_recalc_interval=1, verbose=True, log_file=None):
        """
        Top-k+ Best method with clustering-based sampling for efficiency.
        IMPROVEMENT: Instead of randomly sampling from the best cluster, this method
        selects the most influential point from the best cluster in each iteration.
        
        Args:
            target_ate: Target ATE value
            epsilon: Tolerance for target ATE
            k: Maximum number of points to remove (if target_ate is None)
            approx: Whether to use approximation for ATE calculation
            use_clustering: Whether to use clustering-based sampling
            n_clusters: Number of clusters (if None, automatically determined)
            samples_per_cluster: Number of samples per cluster
            influence_recalc_interval: How often to recalculate influence scores (1=every iteration, 
                                    100=every 100 iterations, 0=only once at start)
            verbose: Whether to log progress information
            log_file: Custom log file name (if None, auto-generated)
            
        Returns:
            Dictionary containing results of the algorithm
        """
        if target_ate is None and k is None:
            raise ValueError("Either target_ate or k must be provided")
        
        # Setup logging
        if verbose:
            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"clustered_top_k_plus_best_{timestamp}.txt"
            
            self._setup_logging(log_file)
            
            self._log(f"=== Clustered Top-k+ Best Algorithm ===")
            self._log(f"Dataset size: {len(self.X)}")
            self._log(f"Use clustering: {use_clustering}")
            if use_clustering:
                self._log(f"Samples per cluster: {samples_per_cluster}")
            self._log(f"Influence recalc interval: {influence_recalc_interval}")
            self._log(f"Model type: {self.model_type}")
            self._log(f"Log file: {self.current_log_file}")
            self._log(f"IMPROVEMENT: Selects best point from best cluster (not random)")
        
        if verbose:
            self._log(f"Original ATE: {self.original_ate}")
            if target_ate:
                self._log(f"Target ATE range: {target_ate - epsilon} to {target_ate + epsilon}")
        
        n_samples = len(self.X)
        
        # Pre-cluster the data once
        if use_clustering:
            if verbose:
                self._log(f"\n--- Step 1: Creating clusters ---")
            
            cluster_mapping, cluster_representatives = self._create_clusters_and_sample(
                n_clusters=n_clusters, samples_per_cluster=samples_per_cluster, verbose=verbose)
            
            # Create cluster membership tracking (simplified)
            clusters = {}  # cluster_id -> set of original indices
            
            # Initialize empty clusters
            for cluster_id in cluster_representatives.keys():
                clusters[cluster_id] = set()
            
            # Assign points to clusters based on their representatives
            for original_idx, rep_idx in cluster_mapping.items():
                # Find which cluster this representative belongs to
                for cluster_id, representatives in cluster_representatives.items():
                    if rep_idx in representatives:
                        clusters[cluster_id].add(original_idx)
                        break
            
            if verbose:
                cluster_sizes = [len(members) for members in clusters.values()]
                self._log(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        
        # Initialize iteration variables
        removed_indices = []
        current_ate = self.original_ate
        available_indices = set(range(n_samples))
        ate_history = [self.original_ate]
        num_removed_history = [0]
        
        # Initialize influence score caching
        cached_cluster_scores = {}
        cached_point_scores = {}
        last_recalc_iteration = 0
        
        # Set random seed for reproducible sampling
        np.random.seed(42)
        
        # Determine stopping condition
        if k is not None:
            max_removals = min(k, n_samples)
            stopping_condition = lambda ate: len(removed_indices) >= max_removals
            if verbose:
                self._log(f"Stopping condition: Remove up to {max_removals} points")
        else:
            stopping_condition = lambda ate: target_ate - epsilon <= ate <= target_ate + epsilon
            if verbose:
                self._log(f"Stopping condition: Reach target ATE within epsilon tolerance")
        
        overshot = False
        prev_desired_direction = 1 if (target_ate > self.original_ate) else -1
        iteration = 0
        
        if verbose:
            self._log(f"\n--- Step 2: Starting iterative removal process ---")
        
        while not stopping_condition(current_ate) and available_indices:
            iteration += 1
            desired_direction = 1 if (target_ate > current_ate) else -1
            
            if desired_direction != prev_desired_direction:
                overshot = True
                if verbose:
                    self._log(f"\nIteration {iteration}: Direction changed - Overshot the target")
                    self._log(f"Current ATE: {current_ate}, Target: {target_ate}")
                    self._log(f"Reversing direction from {prev_desired_direction} to {desired_direction} to move back toward target")
            
            prev_desired_direction = desired_direction
            
            if verbose and iteration == 1:
                if desired_direction == 1:
                    self._log(f"Initial direction: Increasing ATE toward target")
                else:
                    self._log(f"Initial direction: Decreasing ATE toward target")
            
            # Determine if we need to recalculate influence scores
            need_recalc = (iteration == 1 or  # Always calculate on first iteration
                        influence_recalc_interval <= 0 or  # Never recalculate if <= 0
                        (influence_recalc_interval > 0 and 
                        (iteration - last_recalc_iteration) >= influence_recalc_interval))
            
            if use_clustering:
                if need_recalc:
                    if verbose:
                        self._log(f"\nIteration {iteration}: Recalculating cluster-level influence scores...")
                    
                    # Calculate cluster-level influence scores by sampling points from each cluster
                    cluster_scores = {}
                    available_clusters = set()
                    
                    for cluster_id in clusters:
                        # Find available points in this cluster
                        available_in_cluster = list(clusters[cluster_id].intersection(available_indices))
                        if not available_in_cluster:
                            continue  # Skip empty clusters
                        
                        available_clusters.add(cluster_id)

                        # Sample up to 20 points from this cluster to calculate influence
                        n_samples_cluster = min(20, len(available_in_cluster))
                        sampled_points = np.random.choice(available_in_cluster, size=n_samples_cluster, replace=False)
                        
                        cluster_influence_scores = []
                        for sample_idx in sampled_points:
                            remain_df = self.df.drop([sample_idx])
                            ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                            cluster_influence_scores.append(ate_diff)
                        
                        # Use average influence of sampled points as cluster score
                        cluster_scores[cluster_id] = np.mean(cluster_influence_scores)
                    
                    # Cache the scores
                    cached_cluster_scores = cluster_scores
                    last_recalc_iteration = iteration
                    
                    if verbose:
                        self._log(f"Calculated influence scores for {len(cluster_scores)} clusters")
                else:
                    # Use cached scores, but filter for available clusters
                    cluster_scores = {}
                    for cluster_id in clusters:
                        available_in_cluster = list(clusters[cluster_id].intersection(available_indices))
                        if available_in_cluster and cluster_id in cached_cluster_scores:
                            cluster_scores[cluster_id] = cached_cluster_scores[cluster_id]
                    
                    if verbose and iteration % 100 == 1:
                        self._log(f"Iteration {iteration}: Using cached cluster influence scores ({len(cluster_scores)} available clusters)")
                
                # Filter clusters that move ATE in the desired direction
                valid_clusters = [(cluster_id, score) for cluster_id, score in cluster_scores.items() if score * desired_direction > 0]
                
                if not valid_clusters:
                    if verbose:
                        self._log(f"Iteration {iteration}: No clusters move ATE in the desired direction, selecting least harmful cluster")
                    valid_clusters = sorted(cluster_scores.items(), key=lambda x: abs(x[1]))
                    if not valid_clusters:
                        if verbose:
                            self._log(f"Iteration {iteration}: No more clusters available. Stopping.")
                        break
                
                # Find the cluster with highest influence in the desired direction
                if valid_clusters[0][1] * desired_direction > 0:
                    best_cluster_id = max(valid_clusters, key=lambda x: abs(x[1]) if x[1] * desired_direction > 0 else 0)[0]
                else:
                    best_cluster_id = valid_clusters[0][0]
                
                # IMPROVEMENT: Instead of randomly sampling, find the BEST point in the best cluster
                available_in_best_cluster = list(clusters[best_cluster_id].intersection(available_indices))
                
                if len(available_in_best_cluster) == 1:
                    # Only one point available, select it
                    next_idx = available_in_best_cluster[0]
                    next_influence = cluster_scores[best_cluster_id]  # Use cluster average as approximation
                else:
                    # Check if we have cached point-level scores for this cluster
                    cluster_cache_key = f"cluster_{best_cluster_id}"
                    
                    if need_recalc or cluster_cache_key not in cached_point_scores:
                        # Calculate influence scores for all available points in the best cluster
                        best_cluster_influences = []
                        for candidate_idx in available_in_best_cluster:
                            remain_df = self.df.drop([candidate_idx])
                            ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                            best_cluster_influences.append((candidate_idx, ate_diff))
                        
                        # Cache the point-level scores for this cluster
                        cached_point_scores[cluster_cache_key] = {idx: diff for idx, diff in best_cluster_influences}
                        
                        if verbose and iteration <= 10:
                            self._log(f"Calculated point-level influence for {len(best_cluster_influences)} points in best cluster")
                    else:
                        # Use cached point-level scores
                        cached_scores = cached_point_scores[cluster_cache_key]
                        best_cluster_influences = [(idx, diff) for idx, diff in cached_scores.items() if idx in available_in_best_cluster]
                        
                        if verbose and iteration <= 10:
                            self._log(f"Using cached point-level influence for {len(best_cluster_influences)} points in best cluster")
                    
                    # Select the point with highest influence in the desired direction
                    valid_points = [(idx, diff) for idx, diff in best_cluster_influences if diff * desired_direction > 0]
                    
                    if valid_points:
                        # Select point with maximum influence in desired direction
                        next_idx, next_influence = max(valid_points, key=lambda x: abs(x[1]))
                    else:
                        # No points move in desired direction, select least harmful
                        next_idx, next_influence = min(best_cluster_influences, key=lambda x: abs(x[1]))
                
                if verbose and iteration <= 10:
                    recalc_status = "RECALCULATED" if need_recalc else "CACHED"
                    self._log(f"Iteration {iteration}: Selected best cluster {best_cluster_id} (avg score: {cluster_scores[best_cluster_id]:.6f} [{recalc_status}])")
                    self._log(f"  -> Selected best point {next_idx} from cluster (influence: {next_influence:.6f})")
            
            else:
                if need_recalc:
                    if verbose:
                        self._log(f"\nIteration {iteration}: Recalculating influence scores for available points...")
                    
                    # Original method: calculate influence for all available points
                    influence_scores = []
                    for i in available_indices:
                        remain_df = self.df.drop([i])
                        ate_diff = self.get_ate_diff(self.treatment, self.outcome,self.confounders,remain_df,self.original_ate)
                        influence_scores.append((i, ate_diff))
                    
                    # Cache the scores
                    cached_point_scores = {idx: diff for idx, diff in influence_scores}
                    last_recalc_iteration = iteration
                    
                    if verbose:
                        self._log(f"Calculated influence scores for {len(influence_scores)} points")
                else:
                    # Use cached scores, but filter for available points
                    influence_scores = [(idx, diff) for idx, diff in cached_point_scores.items() if idx in available_indices]
                    
                    if verbose and iteration % 100 == 1:
                        self._log(f"Iteration {iteration}: Using cached influence scores ({len(influence_scores)} available points)")
                
                # Filter points that move ATE in the desired direction
                valid_influences = [(idx, diff) for idx, diff in influence_scores if diff * desired_direction > 0]
                
                if not valid_influences:
                    if verbose:
                        self._log(f"Iteration {iteration}: No points move ATE in the desired direction, selecting least harmful point")
                    valid_influences = sorted(influence_scores, key=lambda x: abs(x[1]))
                    if not valid_influences:
                        if verbose:
                            self._log(f"Iteration {iteration}: No more points available. Stopping.")
                        break
                
                # Find the point with highest influence in the desired direction
                if valid_influences[0][1] * desired_direction > 0:
                    next_idx = max(valid_influences, key=lambda x: abs(x[1]) if x[1] * desired_direction > 0 else 0)[0]
                    next_influence = next((diff for idx, diff in valid_influences if idx == next_idx), 0)
                else:
                    next_idx = valid_influences[0][0]
                    next_influence = valid_influences[0][1]
                
                if verbose and iteration <= 10:
                    recalc_status = "RECALCULATED" if need_recalc else "CACHED"
                    self._log(f"Iteration {iteration}: Selected point {next_idx} (influence: {next_influence:.6f} [{recalc_status}])")
            
            # Remove the selected point
            removed_indices.append(next_idx)
            available_indices.remove(next_idx)
            
            # Update the model and get new ATE
            self.df = self.df.drop([next_idx])
            current_ate = self.calc_ate(self.treatment, self.outcome,self.confounders, self.df)

            ate_history.append(current_ate)
            num_removed_history.append(len(removed_indices))
            
            if verbose and len(removed_indices) % 500 == 0:
                self._log(f"Iteration {iteration}: Removed {len(removed_indices)} points, Current ATE: {current_ate}")
            
            if stopping_condition(current_ate):
                if verbose:
                    self._log(f"Iteration {iteration}: Stopping condition met!")
                break
        
        result = {
            'removed_indices': removed_indices,
            'final_ate': current_ate,
            'num_removed': len(removed_indices),
            'original_ate': self.original_ate,
            'ate_difference': current_ate - self.original_ate,
            'ate_history': ate_history,
            'num_removed_history': num_removed_history,
            'use_clustering': use_clustering,
            'computational_efficiency': (len(cluster_representatives) if use_clustering else n_samples) / n_samples,
            'influence_recalc_interval': influence_recalc_interval,
            'total_influence_recalculations': (iteration - 1) // max(1, influence_recalc_interval) + 1 if influence_recalc_interval > 0 else 1,
            'log_file': self.current_log_file if verbose else None,
            'method': 'top_k_plus_best'  # Add method identifier
        }
        
        if verbose:
            self._log(f"\n=== Final Results ===")
            self._log(f"Original ATE: {self.original_ate}")
            self._log(f"Final ATE: {current_ate}")
            self._log(f"ATE Difference: {current_ate - self.original_ate}")
            self._log(f"Removed {len(removed_indices)} data points")
            self._log(f"Influence recalculations: {result['total_influence_recalculations']} times")
            if use_clustering:
                self._log(f"Computational efficiency: {result['computational_efficiency']:.1%}")
            
            self._close_logging()
        
        return result
