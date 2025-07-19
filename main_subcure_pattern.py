from subcure_pattern import main
import pandas as pd

if __name__=="__main__":
    csv_name = ""
    treatment = ""
    outcome = ""
    confounders = []
    desired_ate = None
    epsilon= None

    # default params - can be changed
    k=1000
    size_threshold=0.2
    approx=False
    weights_optimization_method = 1 # 0- no optimization, 1- sorting, 2- real weights
    model_type = "linear"
    df = pd.read_csv(csv_name)
    attributes_for_random_walks = [c for c in df.columns if c not in [treatment, outcome, "Unnamed: 0"]]
    main(csv_name, attributes_for_random_walks, confounders, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method, epsilon, approx, model_type)
