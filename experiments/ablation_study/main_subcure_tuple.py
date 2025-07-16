from clustered_top_k import ClusteredATEBaselines
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
    influence_recalc_interval=10
    
    df = pd.read_csv(csv_name)
    sampled_baselines = ClusteredATEBaselines(df[confounders], df[treatment], df[outcome], model_type=model_type)
    sampled_topk_results = sampled_baselines.top_k_plus(target_ate=desired_ate, epsilon=epsilon, approx=approx, influence_recalc_interval=influence_recalc_interval)
