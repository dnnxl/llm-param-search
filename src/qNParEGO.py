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
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from unsloth import FastModel
import evaluate

from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf_list
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples

# =========================================================
# USER CONFIG
# =========================================================
MODEL_NAME = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
DATA_PATH = "/home/danny.xie/data/dxie/llm-param-search/dataset/FEINA_test_split_train_2.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_BATCH = 10
MC_SAMPLES = 128
BATCH_SIZE = 2  # number of candidates per BO iteration
REF_POINT = torch.tensor([20.0, 0.75], dtype=torch.double)

# =========================================================
# EXPERIMENT FOLDER
# =========================================================
EXPERIMENT_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"/home/danny.xie/data/dxie/llm-param-search/src/bo_results_{EXPERIMENT_ID}"
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Results will be saved in: {RESULTS_DIR}")

# =========================================================
# LOAD MODEL + METRICS
# =========================================================
print("Loading model...")
llm_model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=8192,
)

print("Loading evaluation metrics...")
sari_metric = evaluate.load("sari")
bertscore_metric = evaluate.load("bertscore")

df_global = pd.read_csv(DATA_PATH)

# =========================================================
# PROMPT
# =========================================================
def build_prompt(text):
    return f"Simplifica el siguiente segmento discursivo:\n\nSegmento complejo: {text}\nSegmento simplificado:"

# =========================================================
# CLEANING
# =========================================================
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

# =========================================================
# GENERATION
# =========================================================
def generate_simplification(text, temperature, top_p, top_k, rep_pen, max_tokens):
    prompt = build_prompt(text)
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=rep_pen,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_assistant_response(decoded)

# =========================================================
# CACHE HASH
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
            "temp": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
            "rep": rep_pen,
            "max_tok": max_tokens,
        }

        hash_id = config_hash(params)
        cache_file = os.path.join(CACHE_DIR, hash_id + ".json")
        gen_file_json = os.path.join(CACHE_DIR, hash_id + "_generations.json")
        gen_file_csv = os.path.join(CACHE_DIR, hash_id + "_generations.csv")

        if os.path.exists(cache_file) and os.path.exists(gen_file_json):
            with open(cache_file) as f:
                metrics = json.load(f)
            print("Loaded cached result:", metrics)
            results.append([metrics["sari"], metrics["bert"]])
            continue

        preds, sources, refs_all = [], [], []
        start_time = time.time()

        for _, row_data in tqdm(df_global.iterrows(), total=len(df_global), leave=False):
            source = str(row_data["Segmento"])
            refs = [
                str(row_data["P_Annette"]),
                str(row_data["P_Fiorella"]),
                str(row_data["P_Sugeily"]),
                str(row_data["P_Vivian"]),
            ]

            pred = generate_simplification(
                source, float(temperature), float(top_p),
                int(top_k), float(rep_pen), max_tokens
            )

            preds.append(pred)
            sources.append(source)
            refs_all.append(refs)

        inference_time = time.time() - start_time

        sari = sari_metric.compute(
            sources=sources,
            predictions=preds,
            references=refs_all
        )["sari"]

        bert = np.mean(
            bertscore_metric.compute(
                predictions=preds,
                references=refs_all,
                lang="other"
            )["f1"]
        )

        metrics = {
            "sari": sari,
            "bert": float(bert),
            "inference_time_sec": inference_time
        }

        with open(cache_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save generations
        generation_data = [
            {"source": s, "prediction": p, "references": r}
            for s, p, r in zip(sources, preds, refs_all)
        ]
        with open(gen_file_json, "w") as f:
            json.dump({"params": params, "generations": generation_data}, f, indent=2, ensure_ascii=False)

        df_gen = pd.DataFrame({
            "source": sources,
            "prediction": preds,
            "ref_1": [r[0] for r in refs_all],
            "ref_2": [r[1] for r in refs_all],
            "ref_3": [r[2] for r in refs_all],
            "ref_4": [r[3] for r in refs_all],
        })
        df_gen.to_csv(gen_file_csv, index=False)

        print("Evaluated:", metrics)
        results.append([sari, bert])

    return torch.tensor(results, dtype=torch.double)

# =========================================================
# SEARCH SPACE
# =========================================================
bounds = torch.tensor([
    [0.1, 0.5,   10, 1.0,  20],
    [1.5, 1.0,  200, 2.0, 200],
], dtype=torch.double)

standard_bounds = torch.zeros_like(bounds)
standard_bounds[1] = 1

# =========================================================
# SAVE FUNCTIONS
# =========================================================
def save_bo_state(iteration, train_x, train_obj, train_obj_true):
    x_unnorm = train_x.cpu().numpy()
    noisy = train_obj.cpu().numpy()
    true = train_obj_true.cpu().numpy()

    rows = []
    for i in range(len(x_unnorm)):
        params = {
            "temp": x_unnorm[i][0],
            "top_p": x_unnorm[i][1],
            "top_k": int(x_unnorm[i][2]),
            "rep": x_unnorm[i][3],
            "max_tok": int(x_unnorm[i][4]),
        }

        cache_file = os.path.join(CACHE_DIR, config_hash(params) + ".json")
        inference_time = None
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                inference_time = json.load(f).get("inference_time_sec")

        rows.append({
            "iteration": iteration,
            "temperature": x_unnorm[i][0],
            "top_p": x_unnorm[i][1],
            "top_k": int(x_unnorm[i][2]),
            "repetition_penalty": x_unnorm[i][3],
            "max_tokens": int(x_unnorm[i][4]),
            "sari_true": true[i][0],
            "bert_true": true[i][1],
            "sari_noisy": noisy[i][0],
            "bert_noisy": noisy[i][1],
            "inference_time_sec": inference_time,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, f"bo_iteration_{iteration}.csv"), index=False
    )

def save_pareto(iteration, train_x, train_obj_true):
    pareto_mask = is_non_dominated(train_obj_true)
    pareto_x = train_x[pareto_mask].cpu().numpy()
    pareto_y = train_obj_true[pareto_mask].cpu().numpy()

    df = pd.DataFrame({
        "temperature": pareto_x[:,0],
        "top_p": pareto_x[:,1],
        "top_k": pareto_x[:,2],
        "repetition_penalty": pareto_x[:,3],
        "max_tokens": pareto_x[:,4],
        "sari": pareto_y[:,0],
        "bert": pareto_y[:,1],
    })
    df.to_csv(os.path.join(RESULTS_DIR, f"pareto_iteration_{iteration}.csv"), index=False)

def save_hypervolume(iteration, train_obj_true):
    hv = Hypervolume(ref_point=REF_POINT)
    pareto_mask = is_non_dominated(train_obj_true)
    hv_value = hv.compute(train_obj_true[pareto_mask])
    with open(os.path.join(RESULTS_DIR, "hypervolume.csv"), "a") as f:
        f.write(f"{iteration},{hv_value}\n")

def save_best_configuration(train_x, train_obj_true):
    pareto_mask = is_non_dominated(train_obj_true)
    pareto_x = train_x[pareto_mask]
    pareto_y = train_obj_true[pareto_mask]
    best_idx = torch.argmax(pareto_y.sum(dim=1))

    result = {
        "temperature": float(pareto_x[best_idx][0]),
        "top_p": float(pareto_x[best_idx][1]),
        "top_k": int(pareto_x[best_idx][2]),
        "repetition_penalty": float(pareto_x[best_idx][3]),
        "max_tokens": int(pareto_x[best_idx][4]),
        "sari": float(pareto_y[best_idx][0]),
        "bert": float(pareto_y[best_idx][1]),
    }

    with open(os.path.join(RESULTS_DIR, "best_configuration.json"), "w") as f:
        json.dump(result, f, indent=4)

# =========================================================
# INITIAL DATA + MODEL
# =========================================================
def generate_initial_data(n=6):
    #train_x = torch.rand(n, bounds.shape[1], dtype=torch.double)
    train_x = draw_sobol_samples(
        bounds=bounds,
        n=n,
        q=1
    ).squeeze(1)

    x_unnorm = unnormalize(train_x, bounds)
    train_obj_true = evaluate_decoding_params(x_unnorm)
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

train_x, train_obj, train_obj_true = generate_initial_data()
mll, model = initialize_model(train_x, train_obj)

with open(os.path.join(RESULTS_DIR, "hypervolume.csv"), "w") as f:
    f.write("iteration,hypervolume\n")
with open(os.path.join(RESULTS_DIR, "bo_timing.csv"), "w") as f:
    f.write("iteration,bo_time_sec\n")

# =========================================================
# BO LOOP: qNParEGO
# =========================================================
for iteration in range(N_BATCH):
    print(f"\n=== BO Iteration {iteration+1} ===")
    fit_gpytorch_mll(mll)

    # Posterior mean
    with torch.no_grad():
        pred = model.posterior(train_x).mean

    # Create q random Chebyshev scalarizations
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(train_obj.shape[-1], dtype=torch.double).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights, Y=pred))
        acq_func = qNoisyExpectedImprovement(
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])),
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)

    # Optimize sequentially
    bo_start = time.time()
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )
    bo_time = time.time() - bo_start

    with open(os.path.join(RESULTS_DIR, "bo_timing.csv"), "a") as f:
        f.write(f"{iteration+1},{bo_time}\n")

    new_x = candidates
    new_x_unnorm = unnormalize(new_x, bounds)

    new_obj_true = evaluate_decoding_params(new_x_unnorm)
    noise_std = torch.tensor([0.5, 0.01])
    new_obj = new_obj_true + noise_std * torch.randn_like(new_obj_true)

    # Append
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    train_obj_true = torch.cat([train_obj_true, new_obj_true])

    # Save
    save_bo_state(iteration+1, train_x, train_obj, train_obj_true)
    save_pareto(iteration+1, train_x, train_obj_true)
    save_hypervolume(iteration+1, train_obj_true)

    # Reinitialize GP
    mll, model = initialize_model(train_x, train_obj)

save_best_configuration(train_x, train_obj_true)

print("\nExperiment completed successfully.")
