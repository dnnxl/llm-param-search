"""
===============================================================================
Bayesian Optimization for LLM Decoding Parameter Search
===============================================================================

Description
-----------
This script performs multi-objective Bayesian Optimization (BO) to automatically
search for optimal decoding hyperparameters of a Large Language Model (LLM)
for a text simplification task.

The optimization aims to maximize:
    1. SARI score (simplification quality metric)
    2. BERTScore (semantic similarity metric)

The system evaluates different decoding configurations, models their
performance using Gaussian Processes (GPs), and iteratively proposes improved
configurations using Expected Hypervolume Improvement.

-------------------------------------------------------------------------------
Optimized Decoding Parameters
-------------------------------------------------------------------------------
The following generation parameters are optimized:

    - temperature           ∈ [0.1, 1.5]
    - top_p                 ∈ [0.5, 1.0]
    - top_k                 ∈ [10, 200]
    - repetition_penalty    ∈ [1.0, 2.0]
    - max_tokens            ∈ [20, 200]

The optimization is performed in normalized [0,1] space and unnormalized
before evaluation.

-------------------------------------------------------------------------------
Objectives
-------------------------------------------------------------------------------
Multi-objective optimization is performed on:

    - SARI      (higher is better)
    - BERTScore (higher is better)

Instead of combining them into a single scalar objective, Pareto optimal
solutions are identified and hypervolume is used to measure progress.

Reference Point for Hypervolume:
    REF_POINT = [20.0, 0.75]

-------------------------------------------------------------------------------
Optimization Procedure
-------------------------------------------------------------------------------
1. Generate initial random decoding configurations
2. Evaluate each configuration on the full dataset
3. Compute SARI and BERTScore
4. Fit independent Gaussian Process models for each objective
5. Use qNoisyExpectedHypervolumeImprovement acquisition function
6. Optimize acquisition function to propose new candidates
7. Evaluate new candidates
8. Update GP models
9. Repeat for N_BATCH iterations

Key Parameters:
    - N_BATCH      : number of BO iterations
    - MC_SAMPLES   : Monte Carlo samples for acquisition estimation
    - q = 2        : number of candidates proposed per iteration

-------------------------------------------------------------------------------
Caching Mechanism
-------------------------------------------------------------------------------
To avoid redundant model evaluations:

    - Each decoding configuration is hashed
    - Results are stored in JSON files under cache/
    - If a configuration was previously evaluated, metrics are loaded
      instead of re-running inference

This significantly reduces computation time.

-------------------------------------------------------------------------------
Saved Outputs
-------------------------------------------------------------------------------
All results are saved in a timestamped directory:

bo_results_TIMESTAMP/
│
├── cache/                       # Cached evaluations per config
├── bo_iteration_X.csv           # All tested configurations per iteration
├── pareto_iteration_X.csv       # Pareto front per iteration
├── hypervolume.csv              # Hypervolume progression
├── bo_timing.csv                # BO optimization time per iteration
└── best_configuration.json      # Final selected configuration

Each bo_iteration_X.csv contains:
    - decoding parameters
    - true objective values
    - noisy objective values
    - inference time (seconds)

-------------------------------------------------------------------------------
Performance Tracking
-------------------------------------------------------------------------------
The script tracks:

    - Inference time per decoding configuration
    - BO optimization time per iteration
    - Pareto front evolution
    - Hypervolume improvement across iterations

-------------------------------------------------------------------------------
Modeling Details
-------------------------------------------------------------------------------
- Separate SingleTaskGP model per objective
- Combined using ModelListGP
- SumMarginalLogLikelihood for training
- SobolQMCNormalSampler for Monte Carlo estimation
- qNoisyExpectedHypervolumeImprovement acquisition

-------------------------------------------------------------------------------
Final Output Selection
-------------------------------------------------------------------------------
After completing all BO iterations:

    - Pareto front is computed
    - The configuration maximizing (SARI + BERTScore) is selected
    - Saved to best_configuration.json

-------------------------------------------------------------------------------
Requirements
-------------------------------------------------------------------------------
- PyTorch
- BoTorch
- GPyTorch
- evaluate
- unsloth
- pandas
- numpy
- tqdm

-------------------------------------------------------------------------------
Reproducibility Notes
-------------------------------------------------------------------------------
- Determinism depends on model sampling and GPU behavior
- No global random seed is fixed by default
- Results may vary slightly between runs
- Cached evaluations ensure repeatability within the same experiment

-------------------------------------------------------------------------------
Use Case
-------------------------------------------------------------------------------
This framework is designed for:
    - Research experiments
    - Decoding hyperparameter tuning
    - Multi-objective LLM evaluation

===============================================================================

bo_results_TIMESTAMP/
 ├── cache/                  # cached evaluations
 ├── bo_iteration_X.csv      # all tested points per iteration
 ├── pareto_iteration_X.csv  # Pareto front per iteration
 ├── hypervolume.csv         # BO progress
 ├── bo_timing.csv           # BO optimization time
 └── best_configuration.json # final best decoding setup
"""
import os
import re
import json
import time
import torch
import random
import hashlib
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from unsloth import FastModel
import evaluate

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import draw_sobol_samples

# =========================================================
# USER CONFIG
# =========================================================
OPTIMIZATION_ALGORITHM = "qEHVI"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" #"unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")
DATA_PATH = "/data/dxie/llm-optimization/dataset/FEINA_test_split_train_30.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_BATCH = 10
N_INIT = 50
MC_SAMPLES = 128
REF_POINT = torch.tensor([20.0, 0.75], dtype=torch.double)

EXPERIMENT_ID = "experiment_01"
SEED = 42

# =========================================================
# REPRODUCIBILITY
# =========================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================================================
# DIRECTORIES
# =========================================================
RESULTS_DIR = f"./bo_results_{SAFE_MODEL_NAME}_{EXPERIMENT_ID}"
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "bo_checkpoint.pt")
HV_LOG_PATH = os.path.join(RESULTS_DIR, "hypervolume_log.txt")

print(f"Results directory: {RESULTS_DIR}")

# =========================================================
# SAVE EXPERIMENT METADATA
# =========================================================
metadata = {
    "algorithm_optimization": OPTIMIZATION_ALGORITHM,
    "model_name": MODEL_NAME,
    "dataset_path": DATA_PATH,
    "n_init": N_INIT,
    "n_batch": N_BATCH,
    "mc_samples": MC_SAMPLES,
    "ref_point": REF_POINT.tolist(),
    "seed": SEED,
}

with open(os.path.join(RESULTS_DIR, "experiment_config.json"), "w") as f:
    json.dump(metadata, f, indent=4)

env_info = {
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "device": DEVICE,
    "platform": platform.platform(),
}

with open(os.path.join(RESULTS_DIR, "environment.json"), "w") as f:
    json.dump(env_info, f, indent=4)

# =========================================================
# LOAD MODEL + METRICS
# =========================================================
print("Loading model...")
llm_model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=8192,
)

print("Loading metrics...")
sari_metric = evaluate.load("sari")
bertscore_metric = evaluate.load("bertscore")

df_global = pd.read_csv(DATA_PATH)

# =========================================================
# PROMPT
# =========================================================
def build_prompt(text):
    return f"Simplifica el siguiente segmento discursivo:\n\nSegmento complejo: {text}\nSegmento simplificado:"

def clean_string(s):
    return re.sub(r"\s+", " ", s or "").strip()

def extract_assistant_response(raw_text):
    if "Qwen" in MODEL_NAME:
        return re.sub(r'^.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    elif "Llama" in MODEL_NAME:
        if re.search(r"\bassistant\b", raw_text, flags=re.IGNORECASE):
            parts = re.split(r"\bassistant[:\s]*\b", raw_text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return clean_string(parts[-1])
    return clean_string(raw_text)

def generate_simplification(text, temperature, top_p, top_k, rep_pen, max_tokens):
    prompt = build_prompt(text)

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(rep_pen),
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_string(extract_assistant_response(decoded))

# =========================================================
# CACHE
# =========================================================
def config_hash(params):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def evaluate_decoding_params(x_unnorm: torch.Tensor):
    results = []

    for row in x_unnorm:
        temperature, top_p, top_k, rep_pen, max_tokens = row.tolist()
        max_tokens = int(max_tokens)

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
            "rep_pen": rep_pen,
            "max_tokens": max_tokens,
        }

        hash_id = config_hash(params)
        cache_file = os.path.join(CACHE_DIR, hash_id + ".json")

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                metrics = json.load(f)
            results.append([metrics["sari"], metrics["bert"]])
            continue

        preds, sources, refs_all = [], [], []

        for _, row_data in tqdm(df_global.iterrows(), total=len(df_global), leave=False):
            source = str(row_data["Segmento"])
            refs = [
                str(row_data["P_Annette"]),
                str(row_data["P_Fiorella"]),
                str(row_data["P_Sugeily"]),
                str(row_data["P_Vivian"]),
            ]

            pred = generate_simplification(
                source, temperature, top_p, top_k, rep_pen, max_tokens
            )

            preds.append(pred)
            sources.append(source)
            refs_all.append(refs)

        sari = sari_metric.compute(
            sources=sources,
            predictions=preds,
            references=refs_all
        )["sari"]

        bert = np.mean(
            bertscore_metric.compute(
                predictions=preds,
                references=refs_all,
                lang="other",
                nthreads=8,
            )["f1"]
        )

        metrics = {
            "sari": sari,
            "bert": float(bert),
            "params": params,
            "predictions": preds,
        }

        with open(cache_file, "w") as f:
            json.dump(metrics, f)

        results.append([sari, bert])

    return torch.tensor(results, dtype=torch.double)

# =========================================================
# SEARCH SPACE
# =========================================================
bounds = torch.tensor([
    [0.1, 0.5, 10, 1.0, 50],
    [1.5, 1.0, 200, 2.0, 200],
], dtype=torch.double)

standard_bounds = torch.zeros_like(bounds)
standard_bounds[1] = 1

# =========================================================
# INITIAL DATA
# =========================================================
def generate_initial_data(n=N_INIT):
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
    train_obj_true = evaluate_decoding_params(train_x)
    noise_std = torch.tensor([0.5, 0.01])
    train_obj = train_obj_true + noise_std * torch.randn_like(train_obj_true)
    return train_x, train_obj, train_obj_true

def initialize_model(train_x, train_obj):
    models = []
    noise_var = torch.tensor([0.25, 0.0001], dtype=torch.double)

    for i in range(train_obj.shape[-1]):
        y = train_obj[:, i:i+1]
        yvar = torch.full_like(y, noise_var[i])
        models.append(SingleTaskGP(train_x, y, yvar))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# =========================================================
# CHECKPOINT
# =========================================================
def save_checkpoint(iteration, train_x, train_obj, train_obj_true):
    torch.save({
        "iteration": iteration,
        "train_x": train_x.cpu(),
        "train_obj": train_obj.cpu(),
        "train_obj_true": train_obj_true.cpu(),
    }, CHECKPOINT_PATH)

def load_checkpoint():
    checkpoint = torch.load(CHECKPOINT_PATH)
    return (
        checkpoint["iteration"],
        checkpoint["train_x"],
        checkpoint["train_obj"],
        checkpoint["train_obj_true"],
    )

# =========================================================
# START / RESUME
# =========================================================
if os.path.exists(CHECKPOINT_PATH):
    print("Resuming experiment...")
    start_iter, train_x, train_obj, train_obj_true = load_checkpoint()
else:
    print("Starting new experiment...")
    start_iter = 0
    train_x, train_obj, train_obj_true = generate_initial_data()

mll, model = initialize_model(train_x, train_obj)

# =========================================================
# BO LOOP
# =========================================================
for iteration in range(start_iter, N_BATCH):
    print(f"\n=== BO Iteration {iteration+1} ===")

    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # ----------------------------------------------------
    # qEHVI acquisition
    # ----------------------------------------------------
    with torch.no_grad():
        pred_mean = model.posterior(train_x).mean
    partitioning = FastNondominatedPartitioning(ref_point=REF_POINT.tolist(), Y=pred_mean)

    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=REF_POINT.tolist(),
        partitioning=partitioning,
        sampler=sampler,
    )

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=256,
    )

    new_x = candidates
    new_x_unnorm = unnormalize(new_x, bounds)

    new_obj_true = evaluate_decoding_params(new_x_unnorm)
    noise_std = torch.tensor([0.5, 0.01])
    new_obj = new_obj_true + noise_std * torch.randn_like(new_obj_true)

    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    train_obj_true = torch.cat([train_obj_true, new_obj_true])

    # Hypervolume logging
    hv = Hypervolume(ref_point=REF_POINT)
    pareto_mask = is_non_dominated(train_obj_true)
    hv_value = hv.compute(train_obj_true[pareto_mask])

    with open(HV_LOG_PATH, "a") as f:
        f.write(f"{iteration+1},{hv_value}\n")

    save_checkpoint(iteration + 1, train_x, train_obj, train_obj_true)

    mll, model = initialize_model(train_x, train_obj)

# =========================================================
# SAVE FINAL RESULTS
# =========================================================
print("Saving final results...")

final_x_unnorm = unnormalize(train_x, bounds)

results_df = pd.DataFrame(
    final_x_unnorm.numpy(),
    columns=["temperature", "top_p", "top_k", "rep_pen", "max_tokens"]
)
results_df["sari"] = train_obj_true[:, 0].numpy()
results_df["bert"] = train_obj_true[:, 1].numpy()
results_df.to_csv(os.path.join(RESULTS_DIR, "all_evaluations.csv"), index=False)

pareto_mask = is_non_dominated(train_obj_true)
pareto_df = results_df[pareto_mask.numpy()]
pareto_df.to_csv(os.path.join(RESULTS_DIR, "pareto_front.csv"), index=False)

print("Experiment completed successfully.")
