import pandas as pd
import os
import sys

from ATE_update import *

class ATEBaselines:
    """
    Implementation of Top-k and Top-k+ baseline methods for ATE modification.
    These methods identify and remove data points to achieve a target ATE range.
    Modified to consider the direction of influence toward the target ATE.
    """
    
    def __init__(self, X, T, Y, model_type='linear', lambda_reg=0.1, max_iter=1000, find_confounders=False):
        # Store the dataset
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        
        # Initialize the appropriate ATE update model
        if model_type.lower() == 'linear':
            self.ate_model = ATEUpdateLinear(X, T, Y, find_confounders=find_confounders)
        elif model_type.lower() == 'logistic':
            self.ate_model = ATEUpdateLogistic(X, T, Y, C=lambda_reg, 
                                              max_iter=max_iter, find_confounders=find_confounders)
        else:
            raise ValueError("model_type must be either 'linear' or 'logistic'")
        
        # Store the original ATE
        self.original_ate = self.ate_model.get_original_ate()
        self.model_type = model_type
    
    def top_k(self, target_ate=None, epsilon=0.01, k=None, approx=False, verbose=True):
        """
        Top-k method: Calculate influence scores for all points once, then remove points 
        that move the ATE closer to the target.
        """
        if target_ate is None and k is None:
            raise ValueError("Either target_ate or k must be provided")
        
        if verbose:
            print(f"Original ATE: {self.original_ate}")
            if target_ate:
                print(f"Target ATE range: {target_ate - epsilon} to {target_ate + epsilon}")
        
        n_samples = len(self.X)
        influence_scores = []
        
        # Determine the initial desired direction of ATE change
        initial_direction = 1 if (target_ate > self.original_ate) else -1
        
        if verbose:
            if initial_direction == 1:
                print(f"Initial direction: Increasing ATE toward target")
            else:
                print(f"Initial direction: Decreasing ATE toward target") 
        
        # Calculate influence score for each data point only once at the beginning
        for i in range(n_samples):
            # Calculate ATE change when removing this data point
            ate_diff = self.ate_model.get_ate_difference([i], approx=approx, update=False) if self.model_type == 'linear' else \
                        self.ate_model.get_ate_difference([i], approx=approx, update=False)
            influence_scores.append((i, ate_diff))
        
        # Create two sorted lists - one for each direction of influence
        increasing_ate = [(i, diff) for i, diff in influence_scores if diff > 0]
        decreasing_ate = [(i, diff) for i, diff in influence_scores if diff < 0]
        
        # Sort by magnitude of influence (largest first)
        increasing_ate.sort(key=lambda x: abs(x[1]), reverse=True)
        decreasing_ate.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Initialize tracking variables
        removed_indices = []
        current_ate = self.original_ate
        ate_history = [self.original_ate]  # Track ATE at each step
        num_removed_history = [0]  # Track number of points removed
        used_indices = set()
        
        # Determine stopping condition
        if k is not None:
            max_removals = min(k, n_samples)
            stopping_condition = lambda ate: len(removed_indices) >= max_removals
        else:
            stopping_condition = lambda ate: target_ate - epsilon <= ate <= target_ate + epsilon
        
        count = 0
        # Run until stopping condition is met
        while not stopping_condition(current_ate) and len(removed_indices) < n_samples:
            # Determine the current desired direction based on current ATE vs target
            current_direction = 1 if (target_ate > current_ate) else -1
            
            # Check if direction has changed (overshot detection)
            if current_direction != initial_direction and verbose:
                print(f"Direction changed: Current ATE: {current_ate}, Target: {target_ate}")
                print(f"Switching from {'increasing' if initial_direction > 0 else 'decreasing'} to {'increasing' if current_direction > 0 else 'decreasing'} ATE")
            
            # Select the appropriate list based on current direction
            candidates = increasing_ate if current_direction > 0 else decreasing_ate
            
            # Find the next unused point with largest influence in correct direction
            next_point = None
            for idx, diff in candidates:
                if idx not in used_indices:
                    next_point = (idx, diff)
                    break
            
            # If no points left in desired direction, try the other direction
            if next_point is None:
                if verbose:
                    print(f"No unused points left that move ATE in desired direction")
                other_candidates = decreasing_ate if current_direction > 0 else increasing_ate
                for idx, diff in other_candidates:
                    if idx not in used_indices:
                        next_point = (idx, diff)
                        if verbose:
                            print(f"Using point that moves ATE in opposite direction (least harmful)")
                        break
            
            # If still no points, we're done
            if next_point is None:
                if verbose:
                    print("No more unused points available. Stopping.")
                break
            
            idx, diff = next_point
            
            # Remove the selected point
            removed_indices.append(idx)
            used_indices.add(idx)
            
            # Update the model and get new ATE
            current_ate = self.ate_model.calculate_updated_ATE([idx], approx=approx) if self.model_type == 'linear' else \
                          self.ate_model.calculate_updated_ate([idx], approx=approx)
            
            if count % 50 == 0 and verbose:
                print(f"Removed {len(removed_indices)} points, Current ATE: {current_ate}")
            count += 1

            # Record the history
            ate_history.append(current_ate)
            num_removed_history.append(len(removed_indices))
            
            if verbose and len(removed_indices) % 5 == 0:
                print(f"Removed {len(removed_indices)} points, Current ATE: {current_ate}")
        
        result = {
            'removed_indices': removed_indices,
            'final_ate': current_ate,
            'num_removed': len(removed_indices),
            'original_ate': self.original_ate,
            'ate_difference': current_ate - self.original_ate,
            'ate_history': ate_history,
            'num_removed_history': num_removed_history
        }
        
        if verbose:
            print(f"Final ATE: {current_ate}")
            print(f"ATE Difference: {current_ate - self.original_ate}")
            print(f"Removed {len(removed_indices)} data points")
        
        return result
    
    def top_k_plus(self, target_ate=None, epsilon=0.01, k=None, approx=False, verbose=True):
        """
        Top-k+ method: Recalculate influence scores after each point removal.
        Modified to select points that move ATE closer to the target.
        Direction reverses if target is overshot.
        """
        if target_ate is None and k is None:
            raise ValueError("Either target_ate or k must be provided")
        
        if verbose:
            print(f"Original ATE: {self.original_ate}")
            if target_ate:
                print(f"Target ATE range: {target_ate - epsilon} to {target_ate + epsilon}")
        
        n_samples = len(self.X)
        removed_indices = []
        current_ate = self.original_ate
        available_indices = set(range(n_samples))
        ate_history = [self.original_ate]  # Track ATE at each step
        num_removed_history = [0]  # Track number of points removed
        
        # Determine stopping condition
        if k is not None:
            max_removals = min(k, n_samples)
            stopping_condition = lambda ate: len(removed_indices) >= max_removals
        else:
            stopping_condition = lambda ate: target_ate - epsilon <= ate <= target_ate + epsilon
        
        # Keep track of whether we've overshot
        overshot = False
        prev_desired_direction = 1 if (target_ate > self.original_ate) else -1
        
        while not stopping_condition(current_ate) and available_indices:
            # Determine the current desired direction based on current ATE vs target
            desired_direction = 1 if (target_ate > current_ate) else -1
            
            # Check if direction has changed (overshot detection)
            if desired_direction != prev_desired_direction:
                overshot = True
                if verbose:
                    print(f"Direction changed: Overshot the target. Current ATE: {current_ate}, Target: {target_ate}")
                    print(f"Reversing direction from {prev_desired_direction} to {desired_direction} to move back toward target")
            
            prev_desired_direction = desired_direction
            
            if verbose and len(removed_indices) == 0:
                if desired_direction == 1:
                    print(f"Initial direction: Increasing ATE toward target")
                else:
                    print(f"Initial direction: Decreasing ATE toward target")
            
            influence_scores = []
            
            # Calculate influence for each remaining data point
            for i in available_indices:
                # Calculate updated ATE with just this point removed
                ate_diff = self.ate_model.get_ate_difference([i], approx=approx, update=False) if self.model_type == 'linear' else \
                        self.ate_model.get_ate_difference([i], approx=approx, update=False)
                influence_scores.append((i, ate_diff))
            
            # Filter points that move ATE in the desired direction
            valid_influences = [(idx, diff) for idx, diff in influence_scores if diff * desired_direction > 0]
            
            # If no points move ATE in the desired direction, pick the one with smallest adverse effect
            if not valid_influences:
                if verbose:
                    print("No points move ATE in the desired direction, selecting least harmful point")
                # Sort by minimum adverse effect (smallest absolute value in wrong direction)
                valid_influences = sorted(influence_scores, key=lambda x: abs(x[1]))
                if not valid_influences:
                    if verbose:
                        print("No more points available. Stopping.")
                    break
            
            # Find the point with highest influence in the desired direction
            if valid_influences[0][1] * desired_direction > 0:
                # If points in desired direction exist, pick highest influence
                next_idx = max(valid_influences, key=lambda x: abs(x[1]) if x[1] * desired_direction > 0 else 0)[0]
            else:
                # Otherwise pick least harmful
                next_idx = valid_influences[0][0]
            
            # Remove the selected point
            removed_indices.append(next_idx)
            available_indices.remove(next_idx)
            
            # Update the model and get new ATE
            current_ate = self.ate_model.calculate_updated_ATE([next_idx], approx=approx) if self.model_type == 'linear' else \
                          self.ate_model.calculate_updated_ate([next_idx], approx=approx)
            
            # Record the history
            ate_history.append(current_ate)
            num_removed_history.append(len(removed_indices))
            
            if verbose and len(removed_indices) % 5 == 0:
                print(f"Removed {len(removed_indices)} points, Current ATE: {current_ate}")
            
            # Check if we've reached the target
            if stopping_condition(current_ate):
                break
        
        result = {
            'removed_indices': removed_indices,
            'final_ate': current_ate,
            'num_removed': len(removed_indices),
            'original_ate': self.original_ate,
            'ate_difference': current_ate - self.original_ate,
            'ate_history': ate_history,
            'num_removed_history': num_removed_history
        }
        
        if verbose:
            print(f"Final ATE: {current_ate}")
            print(f"ATE Difference: {current_ate - self.original_ate}")
            print(f"Removed {len(removed_indices)} data points")
        
        return result
