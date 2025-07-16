from clustered_top_k import ClusteredATEBaselines
from batch_sampled_top_k import BatchSampledATEBaselines
import pandas as pd

if __name__=="__main__":
    # params
    csv_name = ""
    treatment = ""
    outcome = ""
    confounders = []
    desired_ate = None
    epsilon= None
    approx = False

    # default params - can be changed
    model_type='linear'
    
    df = pd.read_csv(csv_name)
    if len(df) > 100000:
        sample_file_path = csv_name.replace(".csv", "_sample.csv")
        batch_size=400
        sample_ratio=0.1
        k_neighbors=100
        sampled_baselines = BatchSampledATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type, sample_ratio=sample_ratio, sample_file_path=sample_file_path, use_cached_sample=True, batch_size=batch_size)
        sampled_topk_results = sampled_baselines.batch_sampled_top_k(target_ate=desired_ate, epsilon=epsilon, k_neighbors=k_neighbors, approx=approx)
    else:
        sampled_baselines = ClusteredATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type)
        sampled_topk_results = sampled_baselines.top_k(target_ate=desired_ate, epsilon=epsilon, approx=approx)

        

