import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load CSV
# ==========================================
csv_path = "/data/dxie/llm-optimization/qNEHVI_unsloth_Qwen3-0.6B-unsloth-bnb-4bit_experiment_02/all_evaluations.csv"   # change to your file
df = pd.read_csv(csv_path)

# Select metrics (change to noisy if needed)
sari = df["sari_true"].values
bert = df["bert_true"].values

points = np.column_stack((sari, bert))

# ==========================================
# 2. Pareto Front (Maximization)
# ==========================================
def pareto_front_max(points):
    n = points.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i
                if (
                    points[j][0] >= points[i][0] and
                    points[j][1] >= points[i][1] and
                    (
                        points[j][0] > points[i][0] or
                        points[j][1] > points[i][1]
                    )
                ):
                    is_pareto[i] = False
                    break
    return is_pareto


mask = pareto_front_max(points)
pareto_points = points[mask]

# ==========================================
# 3. Plot Pareto Front
# ==========================================
plt.figure(figsize=(8, 6))

# Plot all runs
plt.scatter(sari, bert, alpha=0.5)
# Plot Pareto front
plt.scatter(pareto_points[:, 0], pareto_points[:, 1])

# Sort Pareto points for line plotting
pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
print(pareto_sorted)
plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1])

plt.xlabel("SARI (maximize)")
plt.ylabel("BERTScore (maximize)")
plt.title("Pareto Front: SARI vs BERTScore")
plt.grid(True)

plt.tight_layout()
plt.savefig("./pareto_plot.png", dpi=300)
plt.show()

# ==========================================
# 4. Save Pareto Configurations
# ==========================================
pareto_configs = df[mask]
pareto_configs.to_csv("./pareto_solutions.csv", index=False)

print("Number of Pareto solutions:", len(pareto_configs))
print("Saved pareto_solutions.csv")
print("Saved pareto_plot.png")

# ==========================================
# 5. Hypervolume (Optional)
# ==========================================
try:
    from pymoo.indicators.hv import HV

    # Reference point should be WORSE than all observed points
    ref_point = np.array([0.0, 0.0])
    hv = HV(ref_point=ref_point)
    hypervolume = hv(pareto_points)

    print("Hypervolume:", hypervolume)

except ImportError:
    print("pymoo not installed — skipping hypervolume.")