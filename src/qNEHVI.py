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
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from unsloth import FastModel
import evaluate

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples


# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-objective BO for LLM decoding optimization")

    parser.add_argument("--experiment_id", type=str, default="experiment_01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimization_algorithm", type=str, default="qNEHVI")

    parser.add_argument("--model_name", type=str,
                        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=8192)

    parser.add_argument("--data_path", type=str,
                        default="/data/dxie/llm-optimization/dataset/FEINA_test_split_train_30.csv")

    parser.add_argument("--n_batch", type=int, default=200)
    parser.add_argument("--n_init", type=int, default=50)
    parser.add_argument("--mc_samples", type=int, default=128)

    parser.add_argument("--ref_sari", type=float, default=20.0)
    parser.add_argument("--ref_bert", type=float, default=0.75)

    parser.add_argument("--noise_sari", type=float, default=0.5)
    parser.add_argument("--noise_bert", type=float, default=0.01)

    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_max", type=float, default=1.5)
    parser.add_argument("--top_p_min", type=float, default=0.5)
    parser.add_argument("--top_p_max", type=float, default=1.0)
    parser.add_argument("--top_k_min", type=int, default=10)
    parser.add_argument("--top_k_max", type=int, default=200)
    parser.add_argument("--rep_pen_min", type=float, default=1.0)
    parser.add_argument("--rep_pen_max", type=float, default=2.0)
    parser.add_argument("--max_tokens_min", type=int, default=50)
    parser.add_argument("--max_tokens_max", type=int, default=200)

    parser.add_argument("--num_restarts", type=int, default=10)
    parser.add_argument("--raw_samples", type=int, default=256)
    parser.add_argument("--candidates_generation", type=int, default=1)

    return parser.parse_args()


args = parse_args()

# =========================================================
# SEED & DEVICE
# =========================================================
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

DEVICE = args.device if torch.cuda.is_available() else "cpu"

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_NAME = args.model_name
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")

REF_POINT = torch.tensor([args.ref_sari, args.ref_bert], dtype=torch.double)
NOISE_SE = torch.tensor([args.noise_sari, args.noise_bert], dtype=torch.double)

bounds = torch.tensor([
    [args.temp_min, args.top_p_min, args.top_k_min, args.rep_pen_min, args.max_tokens_min],
    [args.temp_max, args.top_p_max, args.top_k_max, args.rep_pen_max, args.max_tokens_max],
], dtype=torch.double)

standard_bounds = torch.zeros_like(bounds)
standard_bounds[1] = 1

# =========================================================
# DIRECTORIES
# =========================================================
RESULTS_DIR = f"./{args.optimization_algorithm}_{SAFE_MODEL_NAME}_{args.experiment_id}"
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "bo_checkpoint.pt")
HV_LOG_PATH = os.path.join(RESULTS_DIR, "hypervolume_log.csv")
EVAL_LOG_PATH = os.path.join(RESULTS_DIR, "evaluation_log.csv")
TIME_LOG_PATH = os.path.join(RESULTS_DIR, "time_log.csv")

TEXT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "generated_texts")
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

if not os.path.exists(EVAL_LOG_PATH):
    with open(EVAL_LOG_PATH, "w") as f:
        f.write("phase,iteration,temperature,top_p,top_k,rep_pen,max_tokens,"
                "sari,bert,llm_time_sec,total_input_tokens,total_output_tokens\n")

if not os.path.exists(TIME_LOG_PATH):
    with open(TIME_LOG_PATH, "w") as f:
        f.write("iteration,optimization_time_sec,llm_total_time_sec\n")

# Initialize HV log
if not os.path.exists(HV_LOG_PATH):
    with open(HV_LOG_PATH, "w") as f:
        f.write("iteration,hypervolume\n")

# =========================================================
# LOAD MODEL
# =========================================================
print("Loading model...")
llm_model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=args.max_seq_length,
)

print("Loading metrics...")
sari_metric = evaluate.load("sari")
bertscore_metric = evaluate.load("bertscore")

df_global = pd.read_csv(args.data_path)

# =========================================================
# UTILITIES
# =========================================================
def build_prompt(text):
    return f"Simplifica el siguiente segmento discursivo:\n\n{text}\n\nSegmento simplificado:"


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
    input_tokens = inputs["input_ids"].shape[1]

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

    output_tokens = outputs.shape[1] - input_tokens

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_string(extract_assistant_response(decoded))

    return cleaned, input_tokens, output_tokens


def config_hash(params):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def evaluate_decoding_params(x_unnorm: torch.Tensor, phase="sobol", iteration=0):
    results = []
    total_llm_time = 0.0

    for row in x_unnorm:
        temperature, top_p, top_k, rep_pen, max_tokens = row.tolist()

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
            "rep_pen": rep_pen,
            "max_tokens": int(max_tokens),
        }

        hash_id = config_hash(params)
        cache_file = os.path.join(CACHE_DIR, hash_id + ".json")

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                metrics = json.load(f)
            results.append([metrics["sari"], metrics["bert"]])
            continue

        preds, sources, refs_all = [], [], []

        # --- Per-text logging containers ---
        inference_times = []
        input_tokens_list = []
        output_tokens_list = []

        llm_start = time.time()

        for _, row_data in tqdm(df_global.iterrows(), total=len(df_global), leave=False):
            source = str(row_data["Segmento"])
            refs = [
                str(row_data["P_Annette"]),
                str(row_data["P_Fiorella"]),
                str(row_data["P_Sugeily"]),
                str(row_data["P_Vivian"]),
            ]

            # ---- Measure inference time per text ----
            inf_start = time.time()
            pred, in_tokens, out_tokens = generate_simplification(source, **params)
            inf_time = time.time() - inf_start

            preds.append(pred)
            sources.append(source)
            refs_all.append(refs)

            inference_times.append(inf_time)
            input_tokens_list.append(in_tokens)
            output_tokens_list.append(out_tokens)

        llm_time = time.time() - llm_start
        total_llm_time += llm_time

        # ---- Measure SARI time ----
        sari_start = time.time()
        sari = sari_metric.compute(
            sources=sources,
            predictions=preds,
            references=refs_all
        )["sari"]
        sari_time = time.time() - sari_start

        # ---- Measure BERTScore time ----
        bert_start = time.time()
        bert = np.mean(
            bertscore_metric.compute(
                predictions=preds,
                references=refs_all,
                lang="other",
                nthreads=8,
            )["f1"]
        )
        bert_time = time.time() - bert_start

        metrics = {
            "sari": sari,
            "bert": float(bert),
            "params": params,
            "llm_time": llm_time,
            "sari_time": sari_time,
            "bert_time": bert_time,
        }

        # ==========================================
        # SAVE GENERATED TEXTS + PER-TEXT METADATA
        # ==========================================
        config_filename = (
            f"temp_{temperature:.3f}_"
            f"top_p_{top_p:.3f}_"
            f"top_k_{int(top_k)}_"
            f"rep_{rep_pen:.3f}_"
            f"max_{int(max_tokens)}.csv"
        )

        config_path = os.path.join(TEXT_OUTPUT_DIR, config_filename)

        df_texts = pd.DataFrame({
            "source_text": sources,
            "generated_text": preds,
            "ref_1": [r[0] for r in refs_all],
            "ref_2": [r[1] for r in refs_all],
            "ref_3": [r[2] for r in refs_all],
            "ref_4": [r[3] for r in refs_all],
            "inference_time_sec": inference_times,
            "input_tokens": input_tokens_list,
            "output_tokens": output_tokens_list,
        })

        df_texts.to_csv(config_path, index=False)

        # ---- Save config-level metrics ----
        with open(cache_file, "w") as f:
            json.dump(metrics, f)

        # ---- LOG EACH CONFIG EVALUATION ----
        with open(EVAL_LOG_PATH, "a") as f:
            f.write(
                f"{phase},{iteration},"
                f"{temperature},{top_p},{int(top_k)},"
                f"{rep_pen},{int(max_tokens)},"
                f"{sari},{bert},"
                f"{llm_time},{sari_time},{bert_time}\n"
            )

        results.append([sari, bert])

    return torch.tensor(results, dtype=torch.double), total_llm_time

# =========================================================
# INITIAL DATA
# =========================================================
def generate_initial_data(n=args.n_init):
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
    train_obj_true, llm_time = evaluate_decoding_params(
        train_x, phase="sobol", iteration=0
    )

    with open(TIME_LOG_PATH, "a") as f:
        f.write(f"0,0,{llm_time}\n")
    train_obj = train_obj_true + NOISE_SE * torch.randn_like(train_obj_true)
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj):
    models = []

    train_x = normalize(train_x, bounds)
    for i in range(train_obj.shape[-1]):
        y = train_obj[:, i:i+1]
        yvar = torch.full_like(y, NOISE_SE[i] ** 2)
        models.append(SingleTaskGP(train_x, y, yvar))

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# =========================================================
# HYPERVOLUME
# =========================================================
def compute_and_log_hv(train_obj_true, iteration):
    #Y = train_obj_true.detach()
    #pareto_mask = is_non_dominated(Y)
    #pareto_Y = Y[pareto_mask]

    bd = DominatedPartitioning(ref_point=REF_POINT, Y=train_obj_true)
    volume = bd.compute_hypervolume().item()

    with open(HV_LOG_PATH, "a") as f:
        f.write(f"{iteration},{volume}\n")

    print(f"Hypervolume at iteration {iteration}: {volume:.6f}")
    return volume

# =========================================================
# BO LOOP
# =========================================================
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    train_x = checkpoint["train_x"]
    train_obj = checkpoint["train_obj"]
    train_obj_true = checkpoint["train_obj_true"]
    start_iter = checkpoint["iteration"]
else:
    train_x, train_obj, train_obj_true = generate_initial_data()
    start_iter = 0
    compute_and_log_hv(train_obj_true, iteration=0)

mll, model = initialize_model(train_x, train_obj)

for iteration in range(start_iter, args.n_batch):

    print(f"\n=== BO Iteration {iteration+1} ===")

    # ---- OPTIMIZATION TIME ----
    opt_start = time.time()

    fit_gpytorch_mll(mll)

    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=REF_POINT.tolist(),
        X_baseline=normalize(train_x, bounds),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples])),
        prune_baseline=True,
    )

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=args.candidates_generation,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
    )

    optimization_time = time.time() - opt_start

    # ---- LLM EVALUATION ----
    new_x = unnormalize(candidates, bounds)

    new_obj_true, llm_time = evaluate_decoding_params(
        new_x, phase="bo", iteration=iteration+1
    )

    new_obj = new_obj_true + NOISE_SE * torch.randn_like(new_obj_true)

    # ---- SAVE TIMES ----
    with open(TIME_LOG_PATH, "a") as f:
        f.write(f"{iteration+1},{optimization_time},{llm_time}\n")

    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    train_obj_true = torch.cat([train_obj_true, new_obj_true])

    compute_and_log_hv(train_obj_true, iteration + 1)

    torch.save({
        "iteration": iteration + 1,
        "train_x": train_x,
        "train_obj": train_obj,
        "train_obj_true": train_obj_true,
    }, CHECKPOINT_PATH)

    mll, model = initialize_model(train_x, train_obj)

# =========================================================
# SAVE ALL FINAL EVALUATIONS
# =========================================================
print("Saving all evaluations...")

# Convert tensors to numpy
train_x_np = train_x.detach().cpu().numpy()
train_obj_np = train_obj.detach().cpu().numpy()
train_obj_true_np = train_obj_true.detach().cpu().numpy()

# Build dataframe
columns_x = ["temperature", "top_p", "top_k", "rep_pen", "max_tokens"]
columns_obj = ["sari_noisy", "bert_noisy"]
columns_obj_true = ["sari_true", "bert_true"]

df_results = pd.DataFrame(
    np.hstack([train_x_np, train_obj_np, train_obj_true_np]),
    columns=columns_x + columns_obj + columns_obj_true
)

# Pareto front
pareto_mask = is_non_dominated(torch.tensor(train_obj_true_np))
df_results["is_pareto"] = pareto_mask.numpy()

# Final hypervolume
bd = DominatedPartitioning(ref_point=REF_POINT, Y=train_obj_true)
final_hv = bd.compute_hypervolume().item()
df_results["final_hypervolume"] = final_hv

# Save CSV
FINAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "all_evaluations.csv")
df_results.to_csv(FINAL_RESULTS_PATH, index=False)

print(f"Saved results to {FINAL_RESULTS_PATH}")
print(f"Final Hypervolume: {final_hv:.6f}")


print("\nOptimization completed.")
