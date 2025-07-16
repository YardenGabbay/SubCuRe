import pandas as pd
import time
from subcure_pattern import main as main_subcure_pattern
from ATE_update import ATEUpdateLinear
from clustered_top_k import ClusteredATEBaselines
from batch_sampled_top_k import BatchSampledATEBaselines
import numpy as np
from collections import defaultdict


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

    frac_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    print("======== subcure-pattern =======")
    for frac in frac_list:
        curr_df = pd.read_csv(csv_name).sample(frac=frac)
        curr_df.to_csv(f"{frac}_{csv_name}")
        orig_ate = ATEUpdateLinear(curr_df[confounders], curr_df[treatment], curr_df[outcome]).get_original_ate()
        desired_ate = orig_ate+ate_offset
        print(f"\n--- Testing with frac: {frac}---")
        main_subcure_pattern(f"{frac}_{csv_name}", attributes_for_random_walks, confounders, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method, epsilon, approx, model_type)


    print("======== Naive-greedy =======")
    for frac in frac_list:
        curr_df = pd.read_csv(f"{frac}_{csv_name}")
        orig_ate = ATEUpdateLinear(curr_df[confounders], curr_df[treatment], curr_df[outcome]).get_original_ate()
        desired_ate = orig_ate+ate_offset
        print(f"\n--- Testing with frac: {frac}---")

        if len(df) > 100000:
            sampled_baselines = BatchSampledATEBaselines(curr_df[confounders], curr_df[treatment], curr_df[outcome], model_type=model_type, sample_ratio=sample_ratio, batch_size=batch_size)
            sampled_topk_results = sampled_baselines.batch_sampled_top_k(target_ate=desired_ate, epsilon=epsilon, k_neighbors=k_neighbors, approx=approx)
        else:
            sampled_baselines = ClusteredATEBaselines(curr_df[confounders], curr_df[treatment], curr_df[outcome], model_type=model_type)
            sampled_topk_results = sampled_baselines.top_k(target_ate=desired_ate, epsilon=epsilon, approx=approx)


    print("======== subcure-tuple =======")
    for frac in frac_list:
        curr_df = pd.read_csv(f"{frac}_{csv_name}")
        orig_ate = ATEUpdateLinear(curr_df[confounders], curr_df[treatment], curr_df[outcome]).get_original_ate()
        desired_ate = orig_ate+ate_offset
        print(f"\n--- Testing with frac: {frac}---")
        if len(df) > 100000:
            sampled_baselines = BatchSampledATEBaselines(curr_df[confounders], curr_df[treatment], curr_df[outcome], model_type=model_type, sample_ratio=sample_ratio, batch_size=batch_size)
            sampled_topk_results = sampled_baselines.batch_sampled_top_k_plus(target_ate=desired_ate, epsilon=epsilon, k_neighbors=k_neighbors, approx=approx, influence_recalc_interval=influence_recalc_interval)
        else:
            sampled_baselines = ClusteredATEBaselines(curr_df[confounders], curr_df[treatment], curr_df[outcome], model_type=model_type)
            sampled_topk_results = sampled_baselines.top_k_plus(target_ate=desired_ate, epsilon=epsilon, approx=approx, influence_recalc_interval=influence_recalc_interval)

        
