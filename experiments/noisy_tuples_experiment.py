import numpy as np
import pandas as pd
from ATE_update import ATEUpdateLinear
from clustered_top_k import ClusteredATEBaselines
import time


if __name__=="__main__":
    noisy_tuples_num = [400,700,1000,1300,1600]
    treatment = "T"
    outcome = "Y"
    confounders = ["X1","X2","X3"]

    for n_noise in noisy_tuples_num:
        print("="*100)
        print(f"n_noise: {n_noise}")
        print("="*100)
        # Set random seed
        np.random.seed(0)

        # Generate clean data (10,000 records)
        n_clean = 10000
        X1_clean = np.random.normal(0, 1, n_clean)
        X2_clean = np.random.normal(0, 1, n_clean)
        X3_clean = np.random.normal(0, 1, n_clean)
        linear_comb_t = 0.5 * X1_clean - 0.3 * X2_clean + 0.7 * X3_clean
        p_treatment = 1 / (1 + np.exp(-linear_comb_t))
        T_clean = np.random.binomial(1, p_treatment)
        Y_clean = 1.0 * X1_clean - 2.0 * X2_clean + 1.5 * X3_clean + 5 * T_clean + np.random.normal(0, 1, n_clean)

        df_clean = pd.DataFrame({
            'X1': X1_clean,
            'X2': X2_clean,
            'X3': X3_clean,
            'T': T_clean,
            'Y': Y_clean,
            'is_noisy': 0  # Mark as clean
        })

        ate_update_obj = ATEUpdateLinear(df_clean[['X1', 'X2', 'X3']], df_clean['T'], df_clean['Y'])
        clean_df_ate = ate_update_obj.get_original_ate()
        print(f"clean df ATE: {clean_df_ate}")

        # Generate 100 noisy records
        X1_noise = np.random.normal(0, 1, n_noise)
        X2_noise = np.random.normal(0, 1, n_noise)
        X3_noise = np.random.normal(0, 1, n_noise)
        T_noise = np.random.binomial(1, 0.5, n_noise)
        Y_noise = 1.0 * X1_noise - 2.0 * X2_noise + 1.5 * X3_noise + 600 * T_noise + np.random.normal(0, 1, n_noise)

        df_noise = pd.DataFrame({
            'X1': X1_noise,
            'X2': X2_noise,
            'X3': X3_noise,
            'T': T_noise,
            'Y': Y_noise,
            'is_noisy': 1  # Mark as noisy
        })

        # Combine clean and noisy data
        df_combined = pd.concat([df_clean, df_noise], ignore_index=True)

        # Shuffle rows
        df_combined = df_combined.sample(frac=1, random_state=1).reset_index(drop=True)
        df_combined.drop(columns=["is_noisy"]).to_csv(f"synthetic_second_{n_noise}.csv", index=False)

        ate_update_obj = ATEUpdateLinear(df_combined[['X1', 'X2', 'X3']], df_combined['T'], df_combined['Y'])
        df_ate = ate_update_obj.get_original_ate()
        print(f"whole df ATE: {df_ate}")

        # Get indices of noisy rows
        noisy_indices = df_combined.index[df_combined['is_noisy'] == 1].tolist()

        new_ate = ate_update_obj.calculate_updated_ATE(noisy_indices)
        print(f"whole df ATE after noisy removal: {new_ate}")

        print("Indices of noisy records:", noisy_indices)  

        target_ate = clean_df_ate
        epsilon=0.001

        print("\nsubcure-pattern-approx")
        baselines = ClusteredATEBaselines(df_combined[confounders], df_combined[treatment], df_combined[outcome], model_type='linear')
        results = baselines.top_k_plus(target_ate=target_ate, epsilon=epsilon, approx=True, verbose=True, influence_recalc_interval=10)

        print("\nsubcure-patternn-exact")
        baselines = ClusteredATEBaselines(df_combined[confounders], df_combined[treatment], df_combined[outcome], model_type='linear')
        results = baselines.top_k_plus(target_ate=target_ate, epsilon=epsilon, approx=False, verbose=True, influence_recalc_interval=10)

        print("\nnaive-greedy-approx")
        baselines = ClusteredATEBaselines(df_combined[confounders], df_combined[treatment], df_combined[outcome], model_type='linear')
        results = baselines.top_k(target_ate=target_ate, epsilon=epsilon, approx=True, verbose=True)

        print("\nnaive-greedy-exact")
        baselines = ClusteredATEBaselines(df_combined[confounders], df_combined[treatment], df_combined[outcome], model_type='linear')
        results = baselines.top_k(target_ate=target_ate, epsilon=epsilon, approx=False, verbose=True)
