import time
import pandas as pd
import random
import warnings
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from dowhy import CausalModel
import random

from datetime import datetime
warnings.filterwarnings("ignore")

def calc_ate(treatment, outcome, confounders, data):
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

def choose_random_key(weights_optimization_method, random_choice, global_key_value_score):
    if weights_optimization_method==0:
        return random.choice(list(random_choice.keys()))
    
    if weights_optimization_method==1:
        keys = list(random_choice.keys())
        scores = []
        for key in keys:
            score = global_key_value_score.get((key, random_choice[key]), 0)
            scores.append((key, score))

        sorted_scores = sorted(scores, key=lambda x: x[1])

        weights = {}
        rank = 0
        last_score = None
        for (key, score) in sorted_scores:
            if score != last_score:
                rank += 1
                last_score = score
            weights[key] = rank
        sum_weight = sum(weights.values())
        normalized_weights = [weights[key] / sum_weight for key in keys]
        return random.choices(keys, weights=normalized_weights, k=1)[0]
    
    if weights_optimization_method==2:
        keys = list(random_choice.keys())
        weights = {}
        for key in keys:
            score = global_key_value_score.get((key, random_choice[key]), 0)
            if score < 0:
                weights[key] = 0
            else:
                weights[key] = score + 1

        sum_weight = sum(weights.values())
        if sum_weight == 0:
            scores = []
            for key in keys:
                score = global_key_value_score.get((key, random_choice[key]), 0)
                scores.append((key, score))

            sorted_scores = sorted(scores, key=lambda x: x[1])

            weights = {}
            rank = 0
            last_score = None
            for (key, score) in sorted_scores:
                if score != last_score:
                    rank += 1
                    last_score = score
                weights[key] = rank
            sum_weight = sum(weights.values())
        normalized_weights = [weights[key] / sum_weight for key in keys]
        return random.choices(keys, weights=normalized_weights, k=1)[0]


def k_random_walks(k, treatment, outcome, df, desired_ate, size_threshold, weights_optimization_method, confounders, valid_subgroups, epsilon, approx):
    global_used_combinations = set()
    global_key_value_score = dict()
    df_ate = calc_ate(treatment,outcome,confounders,df)

    print(f"df ATE: {df_ate}")
    df_shape = df.shape[0]

    desired_diff = desired_ate-df_ate
    print(f"dsired ATE: {desired_ate}")
    print(f"dsired diff: {desired_diff}")
    if len(valid_subgroups) < k:
        print(f"Number of leaves is: {len(valid_subgroups)}. Replacing k")
        k = len(valid_subgroups)
    random_choices = random.sample(valid_subgroups, k)

    diff_values = []

    for walk_idx, random_choice in enumerate(random_choices):
        print(f"\nwalk idx: {walk_idx}")

        combo_to_remove = []
        key_value = []
        random_choice = {k:float(v) for k,v in random_choice.items()}
        while len(random_choice) > 0:
            already_exist = False
            combo_hash = frozenset(random_choice.items())

            if combo_hash in global_used_combinations:
                already_exist = True 
            else:
                global_used_combinations.add(combo_hash)

            combo_to_remove.append((random_choice.copy(), already_exist))

            random_key = choose_random_key(weights_optimization_method, random_choice, global_key_value_score)
            random_value = random_choice[random_key]
            key_value.append((random_key, random_value))
            del random_choice[random_key]

        dfs_to_remove_data = []
        df_to_remove = None

        for key, value in reversed(key_value):
            if df_to_remove is None:
                df_to_remove = df[df[key] == value]
            else:
                df_to_remove = df_to_remove[df_to_remove[key] == value]
            dfs_to_remove_data.append((list(df_to_remove.index), df_to_remove.shape[0]))

        tuples_removed_num = set()
        calc_idx = 0

        for i, (df_to_remove_index, df_to_remove_shape) in enumerate(reversed(dfs_to_remove_data)):

            if df_to_remove_shape / df_shape > size_threshold:
                print(f"More than {size_threshold*100}% of tuples. Breaking")
                break

            if df_to_remove_shape in tuples_removed_num:
                print(f"Combo to remove: {combo_to_remove[i][0]}, ATE already computed, tuples_removed: {df_to_remove_shape}")
            elif combo_to_remove[i][1] == True:
                print(f"Combo to remove: {combo_to_remove[i][0]} already exist in other random walk")
            else:
                remain_df = df.drop(df_to_remove_index)
                ate = calc_ate(treatment,outcome,confounders,remain_df)
                diff = ate-df_ate
                diff_values.append(diff)
                print(f"Combo to remove: {combo_to_remove[i][0]}, ATE: {ate}, tuples_removed: {df_to_remove_shape}, diff: {diff}")
                
                if i > 0:
                    if (desired_diff > 0 and ate > prev_ate) or (desired_diff < 0 and ate < prev_ate):
                        global_key_value_score[key_value[i-1]] = global_key_value_score.get(key_value[i-1], 0) + 1
                    else:
                        global_key_value_score[key_value[i-1]] = global_key_value_score.get(key_value[i-1], 0) - 1
                prev_ate = ate

                if ate >= desired_ate-epsilon and ate <= desired_ate+epsilon:

                    print(f"Desired ATE condition met. Exiting after {walk_idx + 1} random walks.")
                    return

                calc_idx += 1

            tuples_removed_num.add(df_to_remove_shape)

    print(f"Desired ATE condition wasn't met.")
    average_diff = np.mean(diff_values)
    variance_diff = np.var(diff_values)
    max_diff = np.max(diff_values)
    min_diff = np.min(diff_values)
    best_diff = min(diff_values, key=lambda x: abs(x - desired_diff))
    t_statistic, p_value = stats.ttest_1samp(diff_values, 0)

    print(f"Average Diff: {average_diff}")
    print(f"Variance of Diff: {variance_diff}")
    print(f"T-test Statistic: {t_statistic}, P-value: {p_value}")
    print(f"Max Diff: {max_diff}")
    print(f"Min Diff: {min_diff}")
    print(f"Best Diff: {best_diff}")

def main(csv_name, attributes_for_random_walks, confounders, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method, epsilon, approx, model_type):
    start_time = time.time()
    df = pd.read_csv(csv_name)
    if len(df) > 100000:
        df = df.sample(frac=0.1).reset_index()
    valid_subgroups = []
    unique_combinations = df[list(attributes_for_random_walks)].drop_duplicates()

    for row in unique_combinations.itertuples(index=False, name=None):
        subgroup_dict = dict(zip(attributes_for_random_walks, row))
        valid_subgroups.append(subgroup_dict)
    cols = list(set(attributes_for_random_walks+confounders+[treatment, outcome]))
    df = df[cols]
    k_random_walks(k, treatment, outcome, df, desired_ate, size_threshold, weights_optimization_method, confounders, valid_subgroups, epsilon, approx, model_type)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
