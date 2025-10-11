import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score
from radar_factory import radar_factory

# === CONFIGURATION ===
MODEL_NAME = "Llama-3.3-70B-Instruct"
PROMPTING_METHOD = "ZS"
CODE_DIR = Path(__file__).resolve().parent
EVAL_DIR = CODE_DIR / "outputs" / f"{MODEL_NAME}_{PROMPTING_METHOD}"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
model = SentenceTransformer(EMBED_MODEL_NAME)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# === LOAD DATA ===
def load_csv(name):
    return pd.read_csv(EVAL_DIR / name)

df_test = load_csv("predictions_test.csv")
# df_train = load_csv("predictions_train.csv")

# === SCORING FUNCTION ===
def score_predictions(df):
    scored_rows = []
    for _, ex in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        ref = str(ex.get("reference", "")).strip()
        pred = str(ex.get("prediction", "")).strip()
        rouge_score = rouge.score(ref, pred)["rougeL"].fmeasure
        ref_emb = model.encode(ref, convert_to_tensor=True)
        pred_emb = model.encode(pred, convert_to_tensor=True)
        cosine_sim = util.cos_sim(ref_emb, pred_emb).item()
        ex["rouge_l"] = rouge_score
        ex["embedding_similarity"] = cosine_sim
        scored_rows.append(ex)
    return pd.DataFrame(scored_rows)

df_test_scored = score_predictions(df_test)
# df_train_scored = score_predictions(df_train)

df_test_scored.to_csv(EVAL_DIR / "scored_predictions_test.csv", index=False)
# df_train_scored.to_csv(EVAL_DIR / "scored_predictions_train.csv", index=False)

# === TOKEN METRICS ===
def compute_token_metrics(df):
    results = []
    for _, ex in tqdm(df.iterrows(), total=len(df), desc="Token-level"):
        pred_tokens = set(str(ex["prediction"]).split())
        ref_tokens = set(str(ex["reference"]).split())
        tp = len(pred_tokens & ref_tokens)
        fp = len(pred_tokens - ref_tokens)
        fn = len(ref_tokens - pred_tokens)
        all_tokens = list(pred_tokens | ref_tokens)
        y_true = [1 if token in ref_tokens else 0 for token in all_tokens]
        y_pred = [1 if token in pred_tokens else 0 for token in all_tokens]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        results.append({
            "input": ex.get("input", ""),
            "prediction": ex.get("prediction", ""),
            "reference": ex.get("reference", ""),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "rouge_l": ex["rouge_l"],
            "embedding_similarity": ex["embedding_similarity"]
        })
    return pd.DataFrame(results)

df_test_metrics = compute_token_metrics(df_test_scored)
# df_train_metrics = compute_token_metrics(df_train_scored)

df_test_metrics.to_csv(EVAL_DIR / "token_level_metrics_test.csv", index=False)
# df_train_metrics.to_csv(EVAL_DIR / "token_level_metrics_train.csv", index=False)

# === HISTOGRAMS (Test Only) ===
for metric in ["precision", "recall", "f1", "jaccard", "rouge_l", "embedding_similarity"]:
    plt.figure()
    plt.hist(df_test_metrics[metric], bins=30, edgecolor='black')
    plt.title(f"{metric.title()} Score Distribution (Test)")
    plt.xlabel(metric.title())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"histogram_{metric}_test.png")
    plt.close()

# === HEATMAP: Precision vs Recall (Test) ===
bins = [i / 10 for i in range(11)]
df_test_metrics["precision_bin"] = pd.cut(df_test_metrics["precision"], bins)
df_test_metrics["recall_bin"] = pd.cut(df_test_metrics["recall"], bins)
heatmap_data = df_test_metrics.groupby(["recall_bin", "precision_bin"], observed=False).size().reset_index(name="count")
pivot_table = heatmap_data.pivot_table(index="recall_bin", columns="precision_bin", values="count", fill_value=0, observed=False)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap="viridis", linewidths=0.5, annot=True, fmt=".0f")
plt.title("Heatmap: Recall vs Precision (Test)")
plt.xlabel("Precision Bin")
plt.ylabel("Recall Bin")
plt.tight_layout()
plt.savefig(EVAL_DIR / "heatmap_precision_vs_recall_test.png")
plt.close()

# === HEATMAP: Embedding Similarity vs ROUGE-L (Test) ===
df_test_metrics["similarity_bin"] = pd.cut(df_test_metrics["embedding_similarity"], bins)
df_test_metrics["rouge_bin"] = pd.cut(df_test_metrics["rouge_l"], bins)
heatmap_sim_data = df_test_metrics.groupby(["rouge_bin", "similarity_bin"], observed=False).size().reset_index(name="count")
pivot_sim_table = heatmap_sim_data.pivot_table(index="rouge_bin", columns="similarity_bin", values="count", fill_value=0, observed=False)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_sim_table, cmap="viridis", linewidths=0.5, annot=True, fmt=".0f")
plt.title("Heatmap: ROUGE-L vs Embedding Similarity (Test)")
plt.xlabel("Embedding Similarity Bin")
plt.ylabel("ROUGE-L Bin")
plt.tight_layout()
plt.savefig(EVAL_DIR / "heatmap_similarity_vs_rouge_test.png")
plt.close()

# === SCATTERPLOTS (Test) ===
plt.figure(figsize=(8, 6))
plt.scatter(df_test_metrics["precision"], df_test_metrics["recall"], alpha=0.5, edgecolor='k', color="mediumvioletred")
plt.title("Scatterplot: Precision vs Recall (Test)")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig(EVAL_DIR / "scatter_precision_vs_recall_test.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df_test_metrics["embedding_similarity"], df_test_metrics["rouge_l"], alpha=0.5, edgecolor='k', color="mediumvioletred")
plt.title("Scatterplot: Embedding Similarity vs ROUGE-L (Test)")
plt.xlabel("Embedding Similarity")
plt.ylabel("ROUGE-L")
plt.grid(True)
plt.tight_layout()
plt.savefig(EVAL_DIR / "scatter_similarity_vs_rouge_test.png")
plt.close()

# === RADAR CHART (Test) ===
labels = ["F1", "Jaccard", "Precision", "Recall", "ROUGE-L", "Similarity"]
values = [
    df_test_metrics["f1"].mean(),
    df_test_metrics["jaccard"].mean(),
    df_test_metrics["precision"].mean(),
    df_test_metrics["recall"].mean(),
    df_test_metrics["rouge_l"].mean(),
    df_test_metrics["embedding_similarity"].mean()
]
theta = radar_factory(len(labels), frame='polygon')
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
ax.plot(theta, values, color="mediumvioletred", linewidth=2)
ax.fill(theta, values, color="mediumvioletred", alpha=0.25)
ax.set_thetagrids([])
label_padding = 1.11
for angle, label in zip(theta, labels):
    ax.text(angle, label_padding, label, size=12, ha='center', va='center')
ax.set_ylim(0, 1.0)
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=45, fontsize=10)
plt.tight_layout()
plt.savefig(EVAL_DIR / "radar_model_metrics_test.png")
plt.close()

# === SUMMARY TABLE (Test) ===
summary_metrics = {
    "F1": df_test_metrics["f1"],
    "Precision": df_test_metrics["precision"],
    "Recall": df_test_metrics["recall"],
    "Jaccard": df_test_metrics["jaccard"],
    "ROUGE-L": df_test_metrics["rouge_l"],
    "Embedding Similarity": df_test_metrics["embedding_similarity"]
}
summary_df = pd.DataFrame({
    "Metric": summary_metrics.keys(),
    "Mean": [s.mean() for s in summary_metrics.values()],
    "Standard Deviation": [s.std() for s in summary_metrics.values()]
})
summary_df.to_csv(EVAL_DIR / "summary_metrics_test.csv", index=False)
