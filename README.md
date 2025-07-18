# SubCuRe

This repository contains the implementation of the causal data repair pipeline SubCuRe, as described in our paper submitted to SIGMOD 2026.

## File Structure

- `datasets/`: All datasets used in experiments
    - German Credit
    - Twins
    - StackOverFlow
    - ACS
- `experiments/`: Experimental scripts
    - `noisy_tuples_experiment.py`: Creates a synthetic data of 10k tuples, inserting 400-1600 noisy tuples and examine subcure results
    - `ablation_study/`: Subcure algorithms without incremental updates optimization
    - `scale_experiment/`: Effect of tuples number, confounders number, and ATE difference on runtime
      
- `ATE_update.py`: Compute and update average - treatment effect (ATE)
- `batch_sampled_top_k.py`: Randomly sampling-based naive greedy and SubCuRe tuple-removal method with batch removal
- `clustered_top_k.py`: Clustering-based Naive Greedy and SubCuRe Tuple-Removal method
- `dataset_sample.py`: Utility functions for sampling large-scale datasets
- `linear_model_unlearning.py`: Certifiable unlearning on linear models
- `main_naive_greedy.py`: Naive greedy baseline for tuple removal
- `main_subcure_pattern.py`: SubCure pattern‐based tuple removal
- `main_subcure_tuple.py`: SubCure tuple‐based tuple removal
- `sampled_top_k.py`: Randomly sampling-based naive greedy and SubCuRe tuple-removal method
- `search_baseline.py`: Basic naive greedy and SubCuRe tuple-removal method without optimization
- `subcure_pattern.py`: Pattern‐based SubCure implementation
- `README.md`: This file

## How to Run
1. **Run the algorithm**
   
    To run a specific experiment, run the command in the control pannel.
    
    Example: to run SubCuRe-Tuple with batch and sampling optimization, you can run:
    ```bash
    python main_subcure_tuple.py
    ```
    and modify the parameters in the python script as you like.

    What should be noticed is that for batch sampled algorithms, we support pre-computation. If you have already sampled a subset from the dataset using our batch-sampled algorithm, you can set the parameter `sample_file_path` to the sampled subset.

2. **Run an experiment**  
   - noisy_tuples_experiment: run the command in the control pannel.
   - ablation_study: modify the parameters in the python script and run the command in the control pannel (main_subcure_tuple.py / main_subcure_pattern.py).
   - scale_experiment: modify the parameters in the python script and run the command in the control pannel (confounders_num.py / tuples_num.py / target_ate.py). Every experiment includes subcure_pattern + subcure_tuple + naive_greedy execution.

4. **Inspect results**  
   Output logs will be saved in the `experiments/logs` directory when setting the parameter `verbose=True` for the algorithm. The results of tuple removed can be directly observed by printing the variables returned by the algorithms. 

## Requirements

- Python 3.7 or higher  
- Install the following (and any others used in your scripts):
  - numpy  
  - pandas  
  - scikit-learn  
  - scipy  
  - matplotlib

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```
