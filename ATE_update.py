import numpy as np
import pandas as pd
from linear_model_unlearning import CertifiableUnlearningLogisticRegression, BaseLinearRegression
from sklearn.linear_model import LogisticRegression
import copy


class ATEUpdateLinear:
    def __init__(self, X, T, Y, find_confounders=False):
        """
        Initialize with dataset and identify confounders using DoWhy.
        
        Parameters:
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Covariates/features
        T : pandas.Series or numpy.ndarray
            Treatment indicator (0 or 1)
        Y : pandas.Series or numpy.ndarray
            Outcome variable
        """
        # Convert inputs to appropriate formats
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        
        if find_confounders:
            # Try to identify confounders using DoWhy
            self.confounders = self._identify_confounders()
            self.confounders = self.confounders if isinstance(self.confounders, list) else self.confounders.get('backdoor')
            # Create design matrix with treatment and confounders
            X_confounders = self.X[self.confounders] if self.confounders else self.X
            self.design_matrix = pd.concat([self.T.reset_index(drop=True), 
                                        X_confounders.reset_index(drop=True)], axis=1)
            column_names = ['treatment'] + (self.confounders if self.confounders else self.X.columns.tolist())
            self.design_matrix.columns = column_names
        else:
            # Use all features as confounders
            intercept = pd.Series(1, index=range(len(self.T)), name='intercept')
            self.design_matrix = pd.concat([intercept, self.T.reset_index(drop=True), self.X.reset_index(drop=True)], axis=1)
            # self.design_matrix = pd.concat([self.T.reset_index(drop=True), self.X.reset_index(drop=True)], axis=1)
            # self.design_matrix.columns = ['treatment'] + self.X.columns.tolist()
            self.design_matrix.columns = ['intercept', 'treatment'] + self.X.columns.tolist()

        # Convert to numpy for faster computation
        self.X_matrix = self.design_matrix.values
        self.Y_matrix = self.Y.values.reshape(-1, 1)

        # Store dimensions
        self.n_samples = self.X_matrix.shape[0]
        self.n_features = self.X_matrix.shape[1]
        
        # Compute initial linear regression
        self.original_model = BaseLinearRegression(self.X_matrix, self.Y_matrix)
        
        # Store original ATE (treatment effect)
        self.original_ate = float(self.original_model.beta[1])
    
    def _identify_confounders(self):
        """
        Use DoWhy to identify confounders.
        
        Returns:
        --------
        list
            List of column names identified as confounders
        """
        try:
            import dowhy
            from dowhy import CausalModel
            import warnings
            warnings.filterwarnings('ignore')  # Suppress DoWhy warnings
            
            # Prepare data
            data = self.X.copy()
            data['treatment'] = self.T.values
            data['outcome'] = self.Y.values
            
            # Create causal graph
            feature_names = self.X.columns.tolist()
            edges = []
            for feat in feature_names:
                edges.append(f"{feat} -> treatment")
                edges.append(f"{feat} -> outcome")
            edges.append("treatment -> outcome")
            
            graph = "digraph {" + "; ".join(edges) + "}"
            
            # Create causal model
            model = CausalModel(
                data=data,
                treatment='treatment',
                outcome='outcome',
                graph=graph,
                approach="backdoor"
            )
            
            # Identify effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Extract confounders
            if hasattr(identified_estimand, 'backdoor_variables') and identified_estimand.backdoor_variables:
                return identified_estimand.backdoor_variables
            else:
                return self.X.columns.tolist()
                
        except ImportError:
            print("DoWhy not installed. Using all variables as potential confounders.")
            return self.X.columns.tolist()
        except Exception as e:
            print(f"Error in confounder identification: {e}. Using all variables.")
            return self.X.columns.tolist()
    
    def get_ate_difference(self, removed_indices, approx=False, update=True):
        """
        Compute the difference in ATE after removing specified data points.
        Permanently updates the model and dataset.
        
        Parameters:
        -----------
        removed_indices : int or list
            Index or indices of data points to remove
        
        Returns:
        --------
        float
            Difference between updated ATE and original ATE
        """
        if not removed_indices:
            return 0.0
        
        # Store current ATE before update
        current_ate = self.original_ate
        
        # Update the model and dataset
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]

        # Extract rows to be removed
        X_remove = self.design_matrix.loc[removed_indices].values
        Y_remove = self.Y.loc[removed_indices].values.reshape(-1, 1)
        
        if approx:
            # Update inverse using Neumann series
            XTX_inv_updated = self.original_model.neumann_update(X_remove)
        else:
            # Update inverse using Woodbury formula
            XTX_inv_updated = self.original_model.woodbury_update(X_remove)

        beta_updated = XTX_inv_updated @ (self.X_matrix.T @ self.Y_matrix - X_remove.T @ Y_remove)
        
        # Update the ATE
        new_ate = float(beta_updated[1])

        if update:
            self.original_model.XTX_inv = XTX_inv_updated
            self.original_model.beta = beta_updated
            self.original_ate = new_ate
        
            keep_indices = [i for i in self.X.index if i not in removed_indices]

            self.X = self.X.loc[keep_indices]
            self.T = self.T.loc[keep_indices]
            self.Y = self.Y.loc[keep_indices]
            
            # Update design matrix if it exists
            if hasattr(self, 'design_matrix'):
                # self.design_matrix = self.design_matrix.loc[keep_indices].reset_index(drop=True)
                self.design_matrix = self.design_matrix.loc[keep_indices]
            
            self.X_matrix = self.design_matrix.values
            self.Y_matrix = self.Y.values.reshape(-1, 1)
            self.n_samples = self.X_matrix.shape[0]
        
        return new_ate - current_ate

    def get_original_ate(self):
        """
        Get the current ATE (treatment effect).
        
        Returns:
        --------
        float
            Current ATE
        """
        return self.original_ate

    def calculate_updated_ATE(self, removed_indices, approx=False):
        """
        Calculate updated ATE after removing specified data points using the Woodbury method.
        Permanently updates the model and dataset.
        
        Parameters:
        -----------
        indices_to_remove : list
            Index or indices of data points to remove
        
        Returns:
        --------
        float
            Updated ATE after removing specified data points
        """
        if not removed_indices:
            return self.original_ate
        
        # Return the ATE difference to maintain backwards compatibility
        self.get_ate_difference(removed_indices, approx=approx, update=True)
        return self.original_ate
    

class ATEUpdateLogistic:
    def __init__(self, X, T, Y, C=0.1, max_iter=1000, find_confounders=False):
        """
        Initialize with dataset and optionally identify confounders using DoWhy.
        
        Parameters:
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Covariates/features
        T : pandas.Series or numpy.ndarray
            Treatment indicator (0 or 1)
        Y : pandas.Series or numpy.ndarray
            Outcome variable
        C : float, default=0.1
            Regularization parameter for logistic regression
        max_iter : int, default=1000
            Maximum number of iterations for optimization
        find_confounders : bool, default=False
            Whether to use DoWhy to identify confounders
        """
        # Convert inputs to appropriate formats
        self.X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        self.T = T.copy() if isinstance(T, pd.Series) else pd.Series(T)
        self.Y = Y.copy() if isinstance(Y, pd.Series) else pd.Series(Y)
        
        # Model parameters
        self.C = C
        self.max_iter = max_iter
        
        if find_confounders:
            # Try to identify confounders using DoWhy
            self.confounders = self._identify_confounders()
            self.confounders = self.confounders if isinstance(self.confounders, list) else self.confounders.get('backdoor')
            # Use identified confounders
            self.X_features = self.X[self.confounders] if self.confounders else self.X
        else:
            # Use all features as confounders (default behavior)
            self.confounders = self.X.columns.tolist()
            self.X_features = self.X
        
        # Initialize and train the original model
        self.original_model = CertifiableUnlearningLogisticRegression(C=self.C, max_iter=self.max_iter)
        self.original_model.fit(self.X_features.values, self.T.values)
        
        # Compute the original ATE
        self.original_ate = self._compute_ate_ipw_unlearning(self.T, self.Y, self.X_features, model=self.original_model)
        
        # Store all available indices
        self.available_indices = list(range(len(self.X)))
    
    def _compute_ate_ipw_unlearning(self, T, Y, X, model=None, removed_index=None, approx=False):
        """
        Compute ATE using Inverse Propensity Weighting with unlearning capability.
        
        Parameters:
        ----------
        T : pandas.Series
            Treatment indicator
        Y : pandas.Series
            Outcome variable
        X : pandas.DataFrame
            Features for propensity score estimation
        model : CertifiableUnlearningLogisticRegression, optional
            Pre-trained model for propensity scores
        removed_index : int or list, optional
            Indices to exclude from ATE calculation
            
        Returns:
        -------
        float
            ATE estimate
        """
        if model is None:
            # Train a new model
            model = CertifiableUnlearningLogisticRegression(C=self.C, max_iter=self.max_iter)
            model.fit(X.values, T.values)
        elif removed_index is not None:
            # Create copies of data without the removed points for unlearning
            if isinstance(removed_index, int):
                removed_index = [removed_index]
                
            X_remove = X.loc[removed_index].values
            T_remove = T.loc[removed_index].values
            
            # Keep data
            keep_indices = [i for i in self.X_features.index if i not in removed_index]
            X_keep = X.loc[keep_indices].values
            T_keep = T.loc[keep_indices].values

            # Apply incremental mini-batch unlearning
            if approx:
                model.fit_incremental_mini_batch(
                    X_keep, T_keep, X_remove, T_remove, sigma=0, batch_size=len(X_remove)
                )
            else:
                model.fit(X_keep, T_keep)

        # Compute propensity scores
        propensity_scores = []
        for i in range(len(X)):
            prob = model.sigmoid(X.iloc[i:i+1].values @ model.theta)
            propensity_scores.append(prob[0])
        propensity_scores = np.array(propensity_scores)
        
        # If we're removing indices, exclude them from ATE calculation
        if removed_index is not None:
            if isinstance(removed_index, int):
                removed_index = [removed_index]
                
            # Create mask for indices to include
            include_mask = np.array([i not in removed_index for i in range(len(X))])
            
            # Apply mask
            T_filtered = T[include_mask]
            Y_filtered = Y[include_mask]
            ps_filtered = propensity_scores[include_mask]
            
            treated_mask = (T_filtered == 1)
            control_mask = (T_filtered == 0)
            
            # For treated units
            if np.sum(treated_mask) > 0:
                weighted_sum_treated = np.sum(Y_filtered[treated_mask] / ps_filtered[treated_mask])
                weight_total_treated = np.sum(1 / ps_filtered[treated_mask])
                weighted_mean_treated = weighted_sum_treated / weight_total_treated
            else:
                weighted_mean_treated = 0
    
            # For control units
            if np.sum(control_mask) > 0:
                weighted_sum_control = np.sum(Y_filtered[control_mask] / (1 - ps_filtered[control_mask]))
                weight_total_control = np.sum(1 / (1 - ps_filtered[control_mask]))
                weighted_mean_control = weighted_sum_control / weight_total_control
            else:
                weighted_mean_control = 0
        else:
            treated_mask = (T == 1)
            control_mask = (T == 0)
            
            # For treated units
            if np.sum(treated_mask) > 0:
                weighted_sum_treated = np.sum(Y[treated_mask] / propensity_scores[treated_mask])
                weight_total_treated = np.sum(1 / propensity_scores[treated_mask])
                weighted_mean_treated = weighted_sum_treated / weight_total_treated
            else:
                weighted_mean_treated = 0
    
            # For control units
            if np.sum(control_mask) > 0:
                weighted_sum_control = np.sum(Y[control_mask] / (1 - propensity_scores[control_mask]))
                weight_total_control = np.sum(1 / (1 - propensity_scores[control_mask]))
                weighted_mean_control = weighted_sum_control / weight_total_control
            else:
                weighted_mean_control = 0

        # ATE estimate
        return weighted_mean_treated - weighted_mean_control
    
    def get_ate_difference(self, removed_indices, approx=False, update=True):
        """
        Compute the difference in ATE after removing specified indices.
        Optionally updates the model and dataset based on update parameter.
        
        Parameters:
        ----------
        removed_indices : int or list
            Index or indices of data points to remove
        method : str, default='unlearning'
            Method to use - 'unlearning' or 'retrain'
        update : bool, default=True
            Whether to permanently update the model and dataset
        
        Returns:
        -------
        float
            Difference between updated ATE and original ATE
        """
        if approx:
            method = 'unlearning'
        else:
            method = 'retrain'

        if not removed_indices:
            return 0.0
            
        # Store the current ATE
        current_ate = self.original_ate
        
        if isinstance(removed_indices, int):
            removed_indices = [removed_indices]
        
        # Create updated model (copy if not updating permanently)
        updated_model = copy.deepcopy(self.original_model)
        # if update:
        #     updated_model = self.original_model
        # else:
        #     # Create a copy for temporary calculation
        #     updated_model = CertifiableUnlearningLogisticRegression(C=self.C, max_iter=self.max_iter)
        #     updated_model.theta = self.original_model.theta.copy()
        #     updated_model.C = self.original_model.C

        if method == 'unlearning':
            # Extract rows to be removed
            X_remove = self.X_features.loc[removed_indices].values
            T_remove = self.T.loc[removed_indices].values
            
            # Keep data
            keep_indices = [i for i in self.X_features.index if i not in removed_indices]
            X_keep = self.X_features.loc[keep_indices].values
            T_keep = self.T.loc[keep_indices].values
            
            # Update the model using unlearning
            updated_model.fit_incremental_mini_batch(
                X_keep, T_keep, X_remove, T_remove, sigma=0, batch_size=len(X_remove)
            )
        
        elif method == 'retrain':
            # Keep data
            X_remove = self.X_features.loc[removed_indices].values
            T_remove = self.T.loc[removed_indices].values
            keep_indices = [i for i in self.X_features.index if i not in removed_indices]
            X_keep = self.X_features.loc[keep_indices].values
            T_keep = self.T.loc[keep_indices].values
            
            # Retrain the model from scratch
            updated_model.fit(X_keep, T_keep)
        
        else:
            raise ValueError("Method must be either 'unlearning' or 'retrain'")
        
        # Calculate new ATE with updated data
        
        keep_indices = [i for i in self.X_features.index if i not in removed_indices]
        X_temp = self.X.loc[keep_indices]
        T_temp = self.T.loc[keep_indices]
        Y_temp = self.Y.loc[keep_indices]
        X_features_temp = self.X_features.loc[keep_indices]
        
        new_ate = self._compute_ate_ipw_unlearning(T_temp, Y_temp, X_features_temp, model=updated_model, approx=approx)

        if update:
            self.X = X_temp
            self.T = T_temp
            self.Y = Y_temp
            self.X_features = X_features_temp
            self.original_model = copy.deepcopy(updated_model)
            self.original_ate = new_ate
            self.available_indices = [idx for i, idx in enumerate(self.available_indices) if i not in removed_indices]
        
        return new_ate - current_ate

    def get_original_ate(self):
        """
        Get the current ATE (treatment effect).
        
        Returns:
        -------
        float
            Current ATE
        """
        return self.original_ate

    def calculate_updated_ate(self, removed_indices, approx=False):
        """
        Calculate updated ATE after removing specified indices.
        Optionally updates the model and dataset based on update parameter.
        
        Parameters:
        ----------
        removed_indices : int or list
            Index or indices of data points to remove
        method : str, default='unlearning'
            Method to use - 'unlearning' or 'retrain'
        update : bool, default=True
            Whether to permanently update the model and dataset
        
        Returns:
        -------
        float
            Updated ATE after removal
        """
        if not removed_indices:
            return self.original_ate
        
        # Update model and dataset, and get new ATE
        self.get_ate_difference(removed_indices, approx=approx, update=True)
        
        return self.original_ate
    
    def _identify_confounders(self):
        """
        Use DoWhy to identify confounders.
        
        Returns:
        -------
        list
            List of column names identified as confounders
        """
        try:
            import dowhy
            from dowhy import CausalModel
            import warnings
            warnings.filterwarnings('ignore')  # Suppress DoWhy warnings
            
            # Prepare data
            data = self.X.copy()
            data['treatment'] = self.T.values
            data['outcome'] = self.Y.values
            
            # Create causal graph
            feature_names = self.X.columns.tolist()
            edges = []
            for feat in feature_names:
                edges.append(f"{feat} -> treatment")
                edges.append(f"{feat} -> outcome")
            edges.append("treatment -> outcome")
            
            graph = "digraph {" + "; ".join(edges) + "}"
            
            # Create causal model
            model = CausalModel(
                data=data,
                treatment='treatment',
                outcome='outcome',
                graph=graph,
                approach="backdoor"
            )
            
            # Identify effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Extract confounders
            if hasattr(identified_estimand, 'backdoor_variables') and identified_estimand.backdoor_variables:
                return identified_estimand.backdoor_variables
            else:
                return self.X.columns.tolist()
                
        except ImportError:
            print("DoWhy not installed. Using all variables as potential confounders.")
            return self.X.columns.tolist()
        except Exception as e:
            print(f"Error in confounder identification: {e}. Using all variables.")
            return self.X.columns.tolist()
