import re
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from unsloth import FastModel
import evaluate

# ==============================
# PROMPT HELPERS
# ==============================
def replace_marker_in_file(file_path, marker, new_text):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.replace(marker, new_text)


def build_few_shot_prompt(text: str, shot="five_shot", seed=1) -> str:
    file_path = f'/data/dxie/llm/prompt/Prompt_1/{shot}/prompt_{seed}.txt'
    marker = 'PREPROCESS_TEXT'
    return replace_marker_in_file(file_path, marker, text)


def build_prompt(text: str) -> str:
    return (
        "Simplifica el siguiente segmento discursivo:\n\n"
        f"Segmento complejo: {text}\n"
        "Segmento simplificado:"
    )

# ==============================
# CLEANING
# ==============================
def clean_string(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def extract_assistant_response(raw_text: str, model_name: str = "") -> str:
    if not raw_text:
        return ""

    text = raw_text.strip()

    if "SmolLM2" in model_name and re.search(r"\bassistant\b", text, flags=re.IGNORECASE):
        parts = re.split(r"\bassistant[:\s]*\b", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return clean_string(parts[-1])

    elif "gpt-oss" in model_name and "assistantfinal" in text:
        m = re.search(r"assistantfinal[:\s]*(.+)$", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return clean_string(m.group(1))

    elif "Llama" in model_name and re.search(r"\bassistant\b", text, flags=re.IGNORECASE):
        parts = re.split(r"\bassistant[:\s]*\b", text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return clean_string(parts[-1])

    elif ("Qwen3" in model_name or "DeepSeek" in model_name) and "</think>" in text:
        pattern = re.compile(r"</think>\s*(.*?)(?=(<think>|$))", re.DOTALL | re.IGNORECASE)
        results = [m.group(1).strip() for m in pattern.finditer(text)]
        if results:
            return clean_string(" ".join(results))

    lines = [
        line for line in text.splitlines()
        if not re.search(r"^\s*(user|system|assistant)\b", line, flags=re.IGNORECASE)
    ]
    return clean_string("\n".join(lines).strip())

# ==============================
# GENERATION
# ==============================
def generate_simplification(
    model,
    tokenizer,
    text,
    device,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    shot_type="zero_shot",
    seed=1,
):
    if shot_type == "five_shot":
        prompt_content = build_few_shot_prompt(text, shot="five_shot", seed=seed)
    else:
        prompt_content = build_prompt(text)

    messages = [{"role": "user", "content": prompt_content}]

    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()

# ==============================
# MODEL LOADING
# ==============================
def load_model(model_name):
    print("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, tokenizer, device

# ==============================
# CONFIG NAME
# ==============================
def config_to_name(config):
    return (
        f"temp{config['temperature']}_"
        f"topp{config['top_p']}_"
        f"topk{config['top_k']}_"
        f"rep{config['repetition_penalty']}"
    )

# ==============================
# EVALUATION RUN
# ==============================
def evaluate_config_run(model, tokenizer, device, model_name, config, run_id, shot_type):
    config_name = config_to_name(config)
    filename = f"{PRED_DIR}/{config_name}_run{run_id}.csv"

    if os.path.exists(filename):
        print(f"      ⏭️  Skipping existing run file: {filename}")
        existing_df = pd.read_csv(filename)
        return existing_df["SARI"].mean()

    df = pd.read_csv(DATA_PATH)
    sari_metric = evaluate.load("sari")
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), leave=False):
        source = str(row["Segmento"])
        refs = [
            str(row["P_Annette"]),
            str(row["P_Fiorella"]),
            str(row["P_Sugeily"]),
            str(row["P_Vivian"]),
        ]

        raw_pred = generate_simplification(
            model, tokenizer, source, device,
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            shot_type=shot_type,
            seed=1 if shot_type == "five_shot" else 1, #(idx % 19) + 1 if shot_type == "five_shot" else 1,
        )

        pred_clean = extract_assistant_response(raw_pred, model_name)

        sari = sari_metric.compute(
            sources=[source],
            predictions=[pred_clean],
            references=[refs],
        )["sari"]

        records.append({
            "source": source,
            "prediction": pred_clean,
            "ref_Annette": refs[0],
            "ref_Fiorella": refs[1],
            "ref_Sugeily": refs[2],
            "ref_Vivian": refs[3],
            "SARI": sari,
        })

    run_df = pd.DataFrame(records)
    run_df.to_csv(filename, index=False)
    return run_df["SARI"].mean()

# ==============================
# HYPERPARAMETER SWEEP
# ==============================
def run_hyperparameter_experiments(model, tokenizer, device, model_name, shot_type):
    summary_path = f"{SUMMARY_DIR}/decoding_summary.csv"

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        completed_configs = {config_to_name(row) for _, row in summary_df.iterrows()}
    else:
        summary_df = pd.DataFrame()
        completed_configs = set()

    new_records = []
    keys = PARAM_GRID.keys()
    combinations = list(product(*PARAM_GRID.values()))

    for values in combinations:
        config = dict(zip(keys, values))
        config_name = config_to_name(config)

        if config_name in completed_configs:
            continue

        run_scores = []
        for run in range(1, N_RUNS + 1):
            mean_sari = evaluate_config_run(
                model, tokenizer, device, model_name, config, run, shot_type
            )
            run_scores.append(mean_sari)

        new_records.append({
            **config,
            "SARI_mean": float(np.mean(run_scores)),
            "SARI_std": float(np.std(run_scores)),
        })

        updated_df = pd.concat([summary_df, pd.DataFrame(new_records)], ignore_index=True)
        updated_df.to_csv(summary_path, index=False)

    return pd.read_csv(summary_path)

# ==============================
# MAIN
# ==============================
def main(model_name, shot_type):
    model, tokenizer, device = load_model(model_name)

    summary_df = run_hyperparameter_experiments(
        model, tokenizer, device, model_name, shot_type
    )

    summary_path = f"{SUMMARY_DIR}/decoding_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.sort_values("SARI_mean", ascending=False).head())

# ==============================
# CONFIG
# ==============================
DATA_PATH = "/data/dxie/llm/FEINA_test.csv"
MAX_NEW_TOKENS = 256
N_RUNS = 10

PARAM_GRID = {
    "temperature": [0.3, 0.7, 1.0],
    "top_p": [0.8, 0.9, 0.95],
    "top_k": [20, 50, 100],
    "repetition_penalty": [1.0, 1.1, 1.2],
}

def clean_model_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    name = name.replace(".", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    return name


if __name__ == "__main__":
    MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
    SHOT_TYPE = "zero_shot"  # "zero_shot" or "five_shot"

    BASE_PRED_DIR = "/data/dxie/llm-optimization/experiments/predictions"
    BASE_SUMMARY_DIR = "/data/dxie/llm-optimization/experiments/summaries"

    model_id = clean_model_name(MODEL_NAME)

    PRED_DIR = os.path.join(BASE_PRED_DIR, SHOT_TYPE, model_id)
    SUMMARY_DIR = os.path.join(BASE_SUMMARY_DIR, SHOT_TYPE, model_id)

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    main(MODEL_NAME, SHOT_TYPE)
