from subcure_pattern import main

if __name__=="__main__":
    csv_name = ""
    treatment = ""
    outcome = ""
    confounders = []
    attributes_for_random_walks = []
    desired_ate = None
    epsilon= None

    # default params - can be changed
    k=1000
    size_threshold=0.2
    approx=False
    weights_optimization_method = 1 # 0- no optimization, 1- sorting, 2- real weights
    model_type = "linear"

    main(csv_name, attributes_for_random_walks, confounders, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method, epsilon, approx, model_type)
