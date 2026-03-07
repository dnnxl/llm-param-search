import os
import re
import json
import torch
import random
import hashlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from unsloth import FastModel
import evaluate

# pymoo
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# =========================================================
# ARGPARSE
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, default="experiment_01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--metric",
                        type=str,
                        default="sari",
                        choices=["sari", "sbert"],
                        help="Optimization metric")

    parser.add_argument("--model_name",
                        type=str,
                        default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")

    parser.add_argument("--max_seq_length", type=int, default=8192)

    parser.add_argument("--data_path",
                        type=str,
                        default="/data/dxie/llm-optimization/dataset/FEINA_test_split_train_30.csv")

    parser.add_argument("--n_gen", type=int, default=200)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.05)

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

    return parser.parse_args()


args = parse_args()


# =========================================================
# SEED
# =========================================================

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

DEVICE = args.device if torch.cuda.is_available() else "cpu"


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = args.model_name
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")

bounds = torch.tensor([
    [args.temp_min, args.top_p_min, args.top_k_min, args.rep_pen_min, args.max_tokens_min],
    [args.temp_max, args.top_p_max, args.top_k_max, args.rep_pen_max, args.max_tokens_max],
], dtype=torch.double)


# =========================================================
# DIRECTORIES
# =========================================================

RESULTS_DIR = f"./cmaes_{args.metric}_{SAFE_MODEL_NAME}_{args.experiment_id}"

CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
GENERATIONS_DIR = os.path.join(RESULTS_DIR, "generations")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GENERATIONS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.pt")
OBJ_LOG_PATH = os.path.join(RESULTS_DIR, "objective_log.csv")

METADATA_PATH = os.path.join(RESULTS_DIR, "metadata.json")

# =========================================================
# NORMALIZATION UTILITIES
# =========================================================

LOWER_BOUNDS = bounds[0]
UPPER_BOUNDS = bounds[1]

def normalize_params(x_real):
    """
    Convert real parameters -> normalized [0,1]
    """
    return (x_real - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)


def denormalize_params(x_norm):
    """
    Convert normalized [0,1] -> real parameter space
    """
    return LOWER_BOUNDS + x_norm * (UPPER_BOUNDS - LOWER_BOUNDS)


def denormalize_population(X):
    """
    Convert a population matrix from normalized space
    to real parameter space.
    """
    X_tensor = torch.tensor(X, dtype=torch.double)
    return denormalize_params(X_tensor)

# =========================================================
# SAVE METADATA
# =========================================================

def save_metadata():

    metadata = {
        "experiment": {
            "experiment_id": args.experiment_id,
            "seed": args.seed,
            "device": DEVICE,
        },
        "model": {
            "model_name": MODEL_NAME,
            "max_seq_length": args.max_seq_length,
        },
        "dataset": {
            "data_path": args.data_path,
            "num_samples": int(len(df_global)),
        },
        "optimization": {
            "algorithm": "CMA-ES",
            "metric": args.metric,
            "n_gen": args.n_gen,
            "pop_size": args.pop_size,
            "sigma": args.sigma,
        },
        "search_space": {
            "temperature": [args.temp_min, args.temp_max],
            "top_p": [args.top_p_min, args.top_p_max],
            "top_k": [args.top_k_min, args.top_k_max],
            "repetition_penalty": [args.rep_pen_min, args.rep_pen_max],
            "max_tokens": [args.max_tokens_min, args.max_tokens_max],
        }
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {METADATA_PATH}")

def extract_assistant_response(raw_text):
    if "Qwen" in MODEL_NAME:
        return re.sub(r'^.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    elif "Llama" in MODEL_NAME:
        if re.search(r"\bassistant\b", raw_text, flags=re.IGNORECASE):
            parts = re.split(r"\bassistant[:\s]*\b", raw_text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return clean_string(parts[-1])
    # -------------------------
    # Phi
    # -------------------------
    elif "phi" in MODEL_NAME.lower():
        # Common Phi chat template patterns
        # 1) assistant:
        # 2) <|assistant|>
        # 3) <assistant>
        pattern = re.compile(
            r"(?:assistant[:\s]*|<\|assistant\|>|<assistant>)\s*(.+)$",
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(raw_text)
        if match:
            return clean_string(match.group(1))
    # -------------------------
    # Mistral (NEW)
    # -------------------------
    elif "mistral" in MODEL_NAME.lower():
        match = re.search(
            r"Segmento simplificado:\s*(.+)$",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            return clean_string(match.group(1))

    return clean_string(raw_text)
    
# =========================================================
# OBJECTIVE LOG
# =========================================================

def log_objective(value, iteration):

    with open(OBJ_LOG_PATH, "a") as f:
        f.write(f"{iteration},{value}\n")

    print(f"Iteration {iteration} objective: {value:.6f}")


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

llm_model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=args.max_seq_length
)

print("Loading metric...")

sari_metric = None
bertscore_metric = None

#if args.metric == "sari":
sari_metric = evaluate.load("sari")

#elif args.metric == "sbert":
bertscore_metric = evaluate.load("bertscore")


df_global = pd.read_csv(args.data_path)
save_metadata()

# =========================================================
# PROMPT
# =========================================================

def build_prompt(text):

    return f"Simplifica el siguiente segmento discursivo:\n\n{text}\n\nSegmento simplificado:"


# =========================================================
# CLEAN
# =========================================================

def clean_string(s):

    return re.sub(r"\s+", " ", s or "").strip()


# =========================================================
# GENERATION
# =========================================================

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
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return clean_string(extract_assistant_response(decoded))


# =========================================================
# PARAMETER FILENAME
# =========================================================

def params_to_filename(params):

    return (
        f"T{params['temperature']:.3f}_"
        f"P{params['top_p']:.3f}_"
        f"K{params['top_k']}_"
        f"RP{params['rep_pen']:.3f}_"
        f"MT{params['max_tokens']}.csv"
    )


# =========================================================
# HASH
# =========================================================

def config_hash(params):

    return hashlib.md5(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()


# =========================================================
# OBJECTIVE
# =========================================================

def evaluate_decoding_params(X, iteration):

    results = []

    for row in X:

        temperature, top_p, top_k, rep_pen, max_tokens = row.tolist()

        params = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "rep_pen": float(rep_pen),
            "max_tokens": int(max_tokens),
        }

        cache_file = os.path.join(
            CACHE_DIR,
            config_hash(params) + ".json"
        )

        if os.path.exists(cache_file):

            with open(cache_file) as f:
                metrics = json.load(f)

            results.append([metrics["value"]])
            continue

        preds = []
        sources = []
        refs_all = []

        for _, row_data in tqdm(df_global.iterrows(),
                                total=len(df_global),
                                leave=False):

            source = str(row_data["Segmento"])

            refs = [
                str(row_data["P_Annette"]),
                str(row_data["P_Fiorella"]),
                str(row_data["P_Sugeily"]),
                str(row_data["P_Vivian"]),
            ]

            pred = generate_simplification(source, **params)

            preds.append(pred)
            sources.append(source)
            refs_all.append(refs)

        # ==============================
        # METRIC
        # ==============================

        #if args.metric == "sari":

        sari = sari_metric.compute(
            sources=sources,
            predictions=preds,
            references=refs_all
        )["sari"]

        sari_list = []
        for src, pred, refs in zip(sources, preds, refs_all):
            score = sari_metric.compute(
                sources=[src],
                predictions=[pred],
                references=[refs],
            )["sari"]
            sari_list.append(score)

        #elif args.metric == "sbert":

        sbert_list = bertscore_metric.compute(
                predictions=preds,
                references=refs_all,
                lang="other",
                nthreads=8,
            )["f1"]

        sbert = np.mean(
            sbert_list
        )

        metrics = {
            "metric": args.metric,
            "sari": float(sari),
            "sbert": float(sbert),
            "params": params,
            "iteration": iteration
        }

        with open(cache_file, "w") as f:
            json.dump(metrics, f)

        # ===============================
        # SAVE GENERATIONS
        # ===============================

        gen_filename = params_to_filename(params)
        gen_path = os.path.join(GENERATIONS_DIR, gen_filename)

        df_out = pd.DataFrame({
            "source": sources,
            "prediction": preds,
            "ref1": [r[0] for r in refs_all],
            "ref2": [r[1] for r in refs_all],
            "ref3": [r[2] for r in refs_all],
            "ref4": [r[3] for r in refs_all],
            "sbert": sbert_list,
            "sari": sari_list
        })

        df_out.to_csv(gen_path, index=False)
        
        if args.metric == "sari":
            objective = sari
        else:
            objective = sbert
        results.append([objective])

    return torch.tensor(results, dtype=torch.double)


# =========================================================
# GLOBAL STORAGE
# =========================================================

train_x = []
train_obj_true = []


# =========================================================
# PYMOO PROBLEM
# =========================================================

class DecodingProblem(Problem):

    def __init__(self):

        super().__init__(
            n_var=5,
            n_obj=1,
            xl=np.zeros(5),
            xu=np.ones(5),
        )

        self.iteration = 0

    def _evaluate(self, X, out, *args, **kwargs):

        self.iteration += 1

        # Convert normalized CMAES output to real parameter space
        X_real = denormalize_population(X)

        obj = evaluate_decoding_params(
            X_real,
            self.iteration
        )

        Y = obj.numpy()

        out["F"] = -Y

        for i in range(len(X)):
            train_x.append(torch.tensor(X[i]))
            train_obj_true.append(torch.tensor(Y[i]))

        Y_tensor = torch.stack(train_obj_true)

        log_objective(Y_tensor.max().item(), self.iteration)

        torch.save(
            {
                "iteration": self.iteration,
                "train_x": torch.stack(train_x),
                "train_obj_true": Y_tensor
            },
            CHECKPOINT_PATH
        )


# =========================================================
# RUN OPTIMIZATION
# =========================================================

problem = DecodingProblem()

x0_real = torch.mean(bounds, dim=0)
x0_norm = normalize_params(x0_real).numpy()

algorithm = CMAES(
    x0=x0_norm,
    sigma=args.sigma,
    pop_size=args.pop_size
)

termination = get_termination(
    "n_gen",
    args.n_gen
)

print("Starting CMA-ES optimization")

res = minimize(
    problem,
    algorithm,
    termination,
    seed=args.seed,
    verbose=True,
    save_history=True
)

history_data = []

for algo in res.history:
    history_data.append({
        "n_gen": algo.n_gen,
        "n_eval": algo.evaluator.n_eval,
        "f_min": algo.opt.get("F")[0][0],
        "sigma": algo.sigma
    })

print("f_min", algo.opt.get("F"))

df = pd.DataFrame(history_data)
df.to_csv(os.path.join(RESULTS_DIR, "cmaes_history.csv"), index=False)

print("Optimization finished")

print("Best parameters:")
print(res.X)

print("Best objective:")
print(-res.F)