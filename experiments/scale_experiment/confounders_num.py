import pandas as pd
from subcure_pattern import main as main_subcure_pattern
from ATE_update import ATEUpdateLinear
from clustered_top_k import ClusteredATEBaselines
from batch_sampled_top_k import BatchSampledATEBaselines
import numpy as np
import random


if __name__ == "__main__":   
    csv_name = ""
    treatment = ""
    outcome = ""
    confounders =[]
    epsilon = None
    ate_offset = None
    approx=False
    model_type = "linear"

    # subcure-pattern
    df = pd.read_csv(csv_name)
    attributes_for_random_walks = [c for c in df.columns if c not in [treatment, outcome]]
    k=1000
    size_threshold=0.2
    weights_optimization_method = 1 # 0- no optimization, 1- sorting, 2- real weights

    # subcure-tuple + naive-greedy
    batch_size=400
    sample_ratio=0.1
    k_neighbors=100
    
    # subcure-tuple
    influence_recalc_interval=10

    all_cols = [c for c in df.columns if c not in [treatment, outcome]]
    num_confounders_to_test = 7

    print("======== subcure-pattern =======")
    remaining_cols = all_cols.copy()
    confounders = []
    for i in range(1, num_confounders_to_test + 1):
        print(f"\n--- Testing with {i} confounder(s) ---")
        new_conf = random.choice([c for c in remaining_cols if c not in confounders])
        confounders.append(new_conf)
        print(f"Using confounders: {confounders}")
        orig_ate = ATEUpdateLinear(df[confounders], df[treatment], df[outcome]).get_original_ate()
        desired_ate = orig_ate+ate_offset
        main_subcure_pattern(csv_name, attributes_for_random_walks, confounders, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method, epsilon, approx, model_type)


    print("======== Naive-greedy =======")
    remaining_cols = all_cols.copy()
    confounders = []
    for i in range(1, num_confounders_to_test + 1):
        print(f"\n--- Testing with {i} confounder(s) ---")
        new_conf = random.choice([c for c in remaining_cols if c not in confounders])
        confounders.append(new_conf)
        print(f"Using confounders: {confounders}")
        orig_ate = ATEUpdateLinear(df[confounders], df[treatment], df[outcome]).get_original_ate()
        desired_ate = orig_ate+ate_offset

        if len(df) > 100000:
            sampled_baselines = BatchSampledATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type, sample_ratio=sample_ratio, batch_size=batch_size)
            sampled_topk_results = sampled_baselines.batch_sampled_top_k(target_ate=desired_ate, epsilon=epsilon, k_neighbors=k_neighbors, approx=approx)
        else:
            sampled_baselines = ClusteredATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type)
            sampled_topk_results = sampled_baselines.top_k(target_ate=desired_ate, epsilon=epsilon, approx=approx)


    print("======== subcure-tuple =======")
    remaining_cols = all_cols.copy()
    confounders = []
    for i in range(1, num_confounders_to_test + 1):
        print(f"\n--- Testing with {i} confounder(s) ---")
        new_conf = random.choice([c for c in remaining_cols if c not in confounders])
        confounders.append(new_conf)
        print(f"Using confounders: {confounders}")
        orig_ate = ATEUpdateLinear(df[confounders], df[treatment], df[outcome]).get_original_ate()
        desired_ate = orig_ate+ate_offset
        if len(df) > 100000:
            sampled_baselines = BatchSampledATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type, sample_ratio=sample_ratio, batch_size=batch_size)
            sampled_topk_results = sampled_baselines.batch_sampled_top_k_plus(target_ate=desired_ate, epsilon=epsilon, k_neighbors=k_neighbors, approx=approx, influence_recalc_interval=influence_recalc_interval)
        else:
            sampled_baselines = ClusteredATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type)
            sampled_topk_results = sampled_baselines.top_k_plus(target_ate=desired_ate, epsilon=epsilon, approx=approx, influence_recalc_interval=influence_recalc_interval)

        
