from typing import Any, List, Dict, Union, Sequence, Tuple, Callable
import json
import re
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import wilcoxon, pearsonr
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from statsmodels.stats.inter_rater import fleiss_kappa
from pingouin import intraclass_corr
import math
import copy


Nested = Union[List[Any], Dict[str, Any]]  # recursive type alias, will be used in the function evaluate_performance_withGPT_AllAblExpr

# Function to extract numeric values from formatted predictions
# Extract numeric values from formatted predictions
def extract_itemized_scores(text):
    text = str(text)  # Ensure input is a string

    # Define regex for each score (e.g., "1a: X", "2: X", etc.)
    pattern = r"(\d+[a-z]?)\s*\([^)]*\):\s*([0-9NA]+)"

    # Check if the text contains multiple lines (tab or newline characters)
    if '\n' in text or '\t' in text:
        #print("Multi-line text detected, using re.findall for extraction...")
        matches = re.findall(pattern, str(text))
        return {match[0]: match[1] for match in matches}  # Multi-line handling

    else:
        #print("Single-line text detected, using re.search for extraction...")
        # Define NIHSS labels in order
        score_labels = ["1a", "1b", "1c", "2", "3", "4", "5a", "5b", "6a", "6b", "7", "8", "9", "10", "11"]
        scores_dict = {}

        # Use regex to extract numbers between each NIHSS label
        for i in range(len(score_labels) - 1):
            current_label = score_labels[i]
            next_label = score_labels[i + 1]

            # Extract the first number appearing between the two labels
            match = re.search(rf"{current_label}\s*\(.*?\):\s*(\d+).*?{next_label}", text)
            if match:
                scores_dict[current_label] = match.group(1)

        # Handle last label separately
        last_label = score_labels[-1]
        match = re.search(rf"{last_label}\s*\(.*?\):\s*(\d+)", text)
        if match:
            scores_dict[last_label] = match.group(1)

        return scores_dict  # Single-line handling

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):  # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, np.generic):  # Convert numpy scalar to Python scalar
        return obj.item()
    return obj  # Return the original object if not NumPy type

def extract_itemized_scores_ReasonAndScore(text):
    # Updated regex pattern to handle varied "Scoring" formats
    pattern = r"(\d+[a-z]?)\s*\(.*?\):\s*\nReasoning:\s*([\s\S]*?)\s*\nScoring:\s*<\[\<?(\d+|NA)\>?\]>?"

    # Extract matches
    matches = re.findall(pattern, text, re.DOTALL)

    scores_dict = {}

    # Process matches with a fallback mechanism
    for match in matches:
        section = match[0]
        reasoning_text = re.sub(r'\s+', ' ', match[1]).strip()  # Normalize spaces
        score_text = match[2].strip()  # Try extracting from regex

        # Fallback: If regex fails to extract a valid score, capture the first number after "Scoring:"
        if not score_text.isdigit() and score_text != "NA":
            scoring_line_match = re.search(r"Scoring:\s*<?\[?(\d+)", text)
            if scoring_line_match:
                score_text = scoring_line_match.group(1)

        # Skip "Total NIH Stroke Scale Score" as it's different
        if section.lower() == "total nih stroke scale score":
            continue

        # Store the extracted data
        scores_dict[section] = {
            "reasoning": reasoning_text,
            "score": score_text
        }
    #print("DEBUG: Extracted scores ->", scores_dict)
    return scores_dict

# Function to map numeric NIHSS scores to 5-class categories
def map_to_label(value):
    """Maps numeric NIHSS scores to 5-class NIHSS categories."""
    if 0 <= value <= 4:
        return '0-4'
    elif 5 <= value <= 10:
        return '5-10'
    elif 11 <= value <= 15:
        return '11-15'
    elif 16 <= value <= 21:
        return '16-21'
    elif 22 <= value:
        return '>=22'
    else:
        return None  # Handle out-of-range cases

# Function to compute Quadratic Weighted Kappa (QWK)
def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# Function to compute Intra-class Correlation Coefficient (ICC (2,1))
def compute_icc(data: pd.DataFrame):
    """
    Expects columns: 'index' (targets), 'variable' (raters), 'value' (ratings).
    Non-numeric or missing ratings are coerced to NaN and dropped.
    """
    # 1 · ensure numeric ratings
    data = data.copy()                              # keep original intact
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])            # drop rows that became NaN

    # 2 · optional: cast grouping cols to category (recommended by Pingouin docs)
    data["index"]    = data["index"].astype("category")
    data["variable"] = data["variable"].astype("category")

    # 3 · compute ICC
    icc_result = intraclass_corr(
        data    = data,
        targets = "index",
        raters  = "variable",
        ratings = "value"
    )

    # 4 · return the ICC(2,1) point estimate
    icc_2_1 = icc_result.set_index("Type").loc["ICC2", "ICC"]
    return icc_2_1


def compute_icc_all(data: pd.DataFrame) -> dict:
    data = data.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])

    # These casts are optional but recommended by Pingouin docs
    data["index"]    = data["index"].astype("category")
    data["variable"] = data["variable"].astype("category")

    icc_result = intraclass_corr(
        data=data, targets="index", raters="variable", ratings="value"
    )

    def _py(v):
        """Convert NumPy scalars Python scalars for JSON safety."""
        if isinstance(v, (np.generic,)):
            return v.item()
        return v

    icc_dict = {
        row["Type"]: {
            "Description": row["Description"],
            "ICC":  _py(row["ICC"]),
            "F":    _py(row["F"]),
            "df1":  _py(row["df1"]),
            "df2":  _py(row["df2"]),
            "pval": _py(row["pval"]),
            "CI95%": tuple(map(_py, row["CI95%"]))
        }
        for _, row in icc_result.iterrows()
    }

    return icc_dict


# Function to compute Mean Squared Error (MSE)
def compute_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Main function to evaluate performance for CoT tasks
def evaluate_performance_ReasonAndScore(log_file_path, task, output_csv):
    """Computes NIHSS evaluation metrics and saves itemized score data."""
    print("We are in evaluate_performance_ReasonAndScore!")
    print("Processing log file:", log_file_path)

    # Load experimental logs
    with open(log_file_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    # Convert logs and original dataset to DataFrame
    graded_df = pd.DataFrame(logs)
    original_df = pd.DataFrame(task.data_TEST)

    # Keep only the necessary columns from original_df (excluding 'gold_NIHSS_breakdown' to avoid duplication)
    #columns_to_keep = [col for col in original_df.columns if col not in ["gold_NIHSS_breakdown","gold_NIHSS_breakdown","gold_input"]]
    columns_to_keep = [col for col in original_df.columns if col not in ["gold_NIHSS_breakdown","gold_input"]]

    # Merge while keeping only gold_NIHSS_breakdown from graded_df
    graded_df = graded_df.merge(original_df[columns_to_keep], on="Sample_ID", suffixes=("_log", "_orig"))

    # Extract itemized NIHSS scores from gold standard and predictions
    graded_df["gold_scores"] = graded_df["gold_NIHSS_breakdown"].apply(extract_itemized_scores)
    graded_df["pred_ReasonAndScore"] = graded_df["bot_response"].apply(extract_itemized_scores_ReasonAndScore) #!!!Different from the original function
    graded_df["pred_scores"] = graded_df["pred_ReasonAndScore"].apply(lambda x: {key: x[key]["score"] for key in x})

    # Define NIHSS item columns
    nihss_items = ['1a', '1b', '1c', '2', '3', '4', '5a', '5b', '6a', '6b', '7', '8', '9', '10', '11']
    gold_itemized_columns = [f"{item}_gold" for item in nihss_items]
    pred_itemized_columns = [f"{item}_pred" for item in nihss_items]

    # Convert extracted dictionary values into individual columns
    for item in nihss_items:
        graded_df[f"{item}_gold"] = graded_df["gold_scores"].apply(
            lambda x: int(x.get(item, 0)) if str(x.get(item, "NA")).isdigit() else None)
        graded_df[f"{item}_pred"] = graded_df["pred_scores"].apply(
            lambda x: int(x.get(item, 0)) if str(x.get(item, "NA")).isdigit() else None)

    # Compute NIHSS total scores (ignoring NA values)
    graded_df["NIHSS_gold_total"] = graded_df[gold_itemized_columns].sum(axis=1, skipna=True)
    graded_df["NIHSS_pred_total"] = graded_df[pred_itemized_columns].sum(axis=1, skipna=True)

    # Compute 5-class NIHSS labels
    graded_df["NIHSS_Class_gold"] = graded_df["NIHSS_gold_total"].apply(map_to_label)
    graded_df["NIHSS_Class_pred"] = graded_df["NIHSS_pred_total"].apply(map_to_label)

    # Compute evaluation metrics
    accuracy = accuracy_score(graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    kappa = cohen_kappa_score(graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    w_stat, w_p_value = wilcoxon(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    p_stat, p_p_value = pearsonr(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])

    #averaging pairwise QWK on 5 classes, for ordinal categorical subjects, using (graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    # Define the correct order of categories
    ordinal_mapping = {"0-4": 0, "5-10": 1, "11-15": 2, "16-21": 3, ">=22": 4}

    # Convert categorical labels to ordinal integers
    graded_df["NIHSS_Class_gold_num"] = graded_df["NIHSS_Class_gold"].map(ordinal_mapping)
    graded_df["NIHSS_Class_pred_num"] = graded_df["NIHSS_Class_pred"].map(ordinal_mapping)

    qwk_score = compute_qwk(graded_df["NIHSS_Class_gold_num"], graded_df["NIHSS_Class_pred_num"])
    #ICC(2,1), for investigating correlation between raters on signle-subject continuous data, on adjudicated scores,
    # using (graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    #icc_data = pd.DataFrame({
    #    "subject": np.repeat(np.arange(len(graded_df)), 2),  # Repeat each subject twice
    #    "rater": np.tile(["gold", "pred"], len(graded_df)),
    #    "score": np.concatenate([graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"]])})
    #Saves the ICC(2,1) data
    icc_data = graded_df[['NIHSS_gold_total', 'NIHSS_pred_total']].melt(ignore_index=False).reset_index()
    icc_data.to_csv(output_csv.replace(".csv", "_iccdata.csv"), index=False)

    ICC = compute_icc(icc_data)
    ICC_all = compute_icc_all(icc_data)
    #MSE, for investigating the difference between the adjudicated scores and the predicted scores,
    # using (graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    MSE = compute_mse(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])

    #normalized MSE
    MSE_rescaled = compute_mse(
        np.array(graded_df["NIHSS_gold_total"]) / 42,
        np.array(graded_df["NIHSS_pred_total"]) / 42
    )

    # Save detailed extracted scores and predictions to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    graded_df.to_csv(output_csv, index=False)

    # Create a structured JSON dictionary
    results_dict = \
        {
            "eval_params":
                {
                    #            "Sample_IDs": graded_df["Sample_ID"].tolist(),
                    "data sample size": len(graded_df["Sample_ID"].tolist()),
                    "base model": task.base_model,
                    "temperature": task.temperature,
                    "max_tokens": task.max_tokens,
                    "task": task.task,
                    #"OneTimePrompting_method": task.OneTimePrompting_method
                    "trial_num": task.trial_num,
                },
            "metrics":
                {
                    "Accuracy": accuracy,
                    "Cohen's Kappa": kappa,
                    "Wilcoxon": {"W-statistic": w_stat, "p-value": w_p_value},
                    "Pearson": {"Correlation": p_stat, "p-value": p_p_value},
                    "QWK": qwk_score,
                    "ICC": ICC,
                    "ICC_all": ICC_all,
                    "MSE": MSE,
                    "MSE_rescaled": MSE_rescaled,
                }
        }

    # Save to a JSON file
    #json_output_path = output_csv.replace(".csv", ".json")  # Change .csv extension to .json
    # Ensure the "metrics" subfolder exists
    metrics_folder = os.path.join(os.path.dirname(output_csv), "metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    # Save JSON inside the "metrics" folder
    json_output_path = os.path.join(metrics_folder, os.path.basename(output_csv).replace(".csv", ".json"))
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(results_dict, json_file, indent=4, default=convert_numpy)

    print(f" Evaluation metrics and Sample_IDs saved as JSON: {json_output_path}")

    # Return the dictionary as before
    return results_dict


def evaluate_performance_withGPT(log_file_path, task, output_csv):
    """Computes NIHSS evaluation metrics and saves itemized score data."""
    #Special case: NYUGPT

    if task.base_model == "NYUGPT" and task.dataset_GPTResults_path is not None:
        graded_df = pd.read_excel(task.dataset_GPTResults_path)
    else:
        print("Processing log file:", log_file_path)

        # Load experimental logs
        with open(log_file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        # Convert logs and original dataset to DataFrame
        graded_df = pd.DataFrame(logs)

    original_df = pd.DataFrame(task.data_TEST)

    # Keep only the necessary columns from original_df (excluding 'gold_NIHSS_breakdown' to avoid duplication)
    #columns_to_keep = [col for col in original_df.columns if col not in ["gold_NIHSS_breakdown","gold_NIHSS_breakdown","gold_input"]]
    columns_to_keep = [col for col in original_df.columns if col not in ["gold_NIHSS_breakdown","gold_input"]]

    # Merge while keeping only gold_NIHSS_breakdown from graded_df
    graded_df = graded_df.merge(original_df[columns_to_keep], on="Sample_ID", suffixes=("_log", "_orig"))

    # Extract itemized NIHSS scores from gold standard and predictions
    if task.base_model == "NYUGPT":
        print("DEBUG: NYUGPT model detected, using extract_itemized_scores...")
        print("graded_df.columns", graded_df.columns)
        graded_df["gold_scores"] = graded_df[task.ColName_GoldStandard_NIHSS_breakdown].apply(extract_itemized_scores)
        graded_df["pred_scores"] = graded_df["gpt_NIHSS"].apply(extract_itemized_scores)
    else:
        graded_df["gold_scores"] = graded_df["gold_NIHSS_breakdown"].apply(extract_itemized_scores)
        graded_df["pred_scores"] = graded_df["bot_response"].apply(extract_itemized_scores)

    # Define NIHSS item columns
    nihss_items = ['1a', '1b', '1c', '2', '3', '4', '5a', '5b', '6a', '6b', '7', '8', '9', '10', '11']
    gold_itemized_columns = [f"{item}_gold" for item in nihss_items]
    pred_itemized_columns = [f"{item}_pred" for item in nihss_items]

    # Convert extracted dictionary values into individual columns
    for item in nihss_items:
        graded_df[f"{item}_gold"] = graded_df["gold_scores"].apply(
            lambda x: int(x.get(item, 0)) if str(x.get(item, "NA")).isdigit() else None)
        graded_df[f"{item}_pred"] = graded_df["pred_scores"].apply(
            lambda x: int(x.get(item, 0)) if str(x.get(item, "NA")).isdigit() else None)

    # Compute NIHSS total scores (ignoring NA values)
    graded_df["NIHSS_gold_total"] = graded_df[gold_itemized_columns].sum(axis=1, skipna=True)
    graded_df["NIHSS_pred_total"] = graded_df[pred_itemized_columns].sum(axis=1, skipna=True)

    # Compute 5-class NIHSS labels
    graded_df["NIHSS_Class_gold"] = graded_df["NIHSS_gold_total"].apply(map_to_label)
    graded_df["NIHSS_Class_pred"] = graded_df["NIHSS_pred_total"].apply(map_to_label)

    # Compute evaluation metrics
    accuracy = accuracy_score(graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    kappa = cohen_kappa_score(graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    w_stat, w_p_value = wilcoxon(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    p_stat, p_p_value = pearsonr(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])

    #averaging pairwise QWK on 5 classes, for ordinal categorical subjects, using (graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    # Define the correct order of categories
    ordinal_mapping = {"0-4": 0, "5-10": 1, "11-15": 2, "16-21": 3, ">=22": 4}

    # Convert categorical labels to ordinal integers
    graded_df["NIHSS_Class_gold_num"] = graded_df["NIHSS_Class_gold"].map(ordinal_mapping)
    graded_df["NIHSS_Class_pred_num"] = graded_df["NIHSS_Class_pred"].map(ordinal_mapping)

    qwk_score = compute_qwk(graded_df["NIHSS_Class_gold_num"], graded_df["NIHSS_Class_pred_num"])
    #ICC(2,1), for investigating correlation between raters on signle-subject continuous data, on adjudicated scores,
    # using (graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    #icc_data = pd.DataFrame({
    #    "subject": np.repeat(np.arange(len(graded_df)), 2),  # Repeat each subject twice
    #    "rater": np.tile(["gold", "pred"], len(graded_df)),
    #    "score": np.concatenate([graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"]])})
    #Saves the ICC(2,1) data
    icc_data = graded_df[['NIHSS_gold_total', 'NIHSS_pred_total']].melt(ignore_index=False).reset_index()
    print("DEBUG: csv data shape ->", output_csv)
    output_csv_icc = output_csv.replace(".csv", "_iccdata.csv")
    print("DEBUG: ICC output path ->", output_csv_icc)
    if not os.path.exists(output_csv_icc):
        # Save the ICC data to CSV if it doesn't already exist
        os.makedirs(os.path.dirname(output_csv_icc), exist_ok=True)
    icc_data.to_csv(output_csv_icc, index=False)

    ICC = compute_icc(icc_data)
    ICC_all = compute_icc_all(icc_data)
    #MSE, for investigating the difference between the adjudicated scores and the predicted scores,
    # using (graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    MSE = compute_mse(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])

    #normalized MSE
    MSE_rescaled = compute_mse(
        np.array(graded_df["NIHSS_gold_total"]) / 42,
        np.array(graded_df["NIHSS_pred_total"]) / 42
    )

    # Save detailed extracted scores and predictions to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    graded_df.to_csv(output_csv, index=False)

    # Create a structured JSON dictionary
    results_dict = \
    {
        "eval_params":
            {
#            "Sample_IDs": graded_df["Sample_ID"].tolist(),
            "data sample size": len(graded_df["Sample_ID"].tolist()),
            "base model":task.base_model,
            "temperature":task.temperature,
            "max_tokens":task.max_tokens,
            "task": task.task,
            # "OneTimePrompting_method": task.OneTimePrompting_method
            "trial_num": task.trial_num,
            },
        "metrics":
            {
        "Accuracy": accuracy,
        "Cohen's Kappa": kappa,
        "Wilcoxon": {"W-statistic": w_stat, "p-value": w_p_value},
        "Pearson": {"Correlation": p_stat, "p-value": p_p_value},
        "QWK": qwk_score,
        "ICC": ICC,
        "ICC_all": ICC_all,
        "MSE": MSE,
        "MSE_rescaled": MSE_rescaled,
            }
    }

    # Save to a JSON file
    #json_output_path = output_csv.replace(".csv", ".json")  # Change .csv extension to .json
    # Ensure the "metrics" subfolder exists
    metrics_folder = os.path.join(os.path.dirname(output_csv), "metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    # Save JSON inside the "metrics" folder
    json_output_path = os.path.join(metrics_folder, os.path.basename(output_csv).replace(".csv", ".json"))
    print("results_dict", results_dict)

    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(results_dict, json_file, indent=4, default=convert_numpy)

    print(f"🔹 Evaluation metrics and Sample_IDs saved as JSON: {json_output_path}")

    # Return the dictionary as before
    return results_dict


def evaluate_performance_withGPT_InputGradedDF(tuple_input_2elem, task, output_csv):
    """Computes NIHSS evaluation metrics and saves itemized score data."""

    graded_df_input = tuple_input_2elem[0]
    prev_concate_key_names = tuple_input_2elem[1]
    output_csv = '/'.join(output_csv.replace('\\', '/').split('/')[:-1] + [prev_concate_key_names]) + '.csv'

    original_df_tmpTEST = pd.DataFrame(task.data_TEST)

    # Keep only the necessary columns from original_df_tmpTEST (excluding 'gold_NIHSS_breakdown' to avoid duplication)
    #columns_to_keep = [col for col in original_df.columns if col not in ["gold_NIHSS_breakdown","gold_NIHSS_breakdown","gold_input"]]
    columns_to_keep = [col for col in original_df_tmpTEST.columns if col not in ["gold_NIHSS_breakdown","gold_input"]]
    print("!!!columns_to_keep:", columns_to_keep)

    # Merge while keeping only gold_NIHSS_breakdown from graded_df
    # merge method: OUTER JOIN!
    graded_df = graded_df_input.merge(
        original_df_tmpTEST[columns_to_keep],
        on="Sample_ID",
        suffixes=("_log", "_orig"),
        how="outer"  # =H this ensures an outer join
    )
    print("!!!shape of graded_df after merging:", graded_df.shape)

    ##########################################################
    ##########################################################
    # Key step: Produce "gold_scores" and "pred_scores" column
    if "reason" not in graded_df["bot_response"].iloc[0].lower(): #Direct Answer case
        # substep1: get gold scores
        try:
            graded_df["gold_scores"] = graded_df["gold_NIHSS_breakdown"].apply(extract_itemized_scores)
        except Exception as e:
            print("Error extracting gold_scores:", e)
            print("graded_df columns:", graded_df.columns)
            raise e
        # substep2: get predicted scores
        graded_df["pred_scores"] = graded_df["bot_response"].apply(extract_itemized_scores)
    else: #Chain-of-Thought case, with Reason and Score
        # substep1: get gold scores
        graded_df["gold_scores"] = graded_df["gold_NIHSS_breakdown"].apply(extract_itemized_scores)
        # substep2: get predicted scores
        graded_df["pred_ReasonAndScore"] = graded_df["bot_response"].apply(
            extract_itemized_scores_ReasonAndScore)  # !!!Different from the original function
        graded_df["pred_scores"] = graded_df["pred_ReasonAndScore"].apply(
            lambda x: {key: x[key]["score"] for key in x})
    ##########################################################
    ##########################################################

    # Define NIHSS item columns
    nihss_items = ['1a', '1b', '1c', '2', '3', '4', '5a', '5b', '6a', '6b', '7', '8', '9', '10', '11']
    gold_itemized_columns = [f"{item}_gold" for item in nihss_items]
    pred_itemized_columns = [f"{item}_pred" for item in nihss_items]

    # Convert extracted dictionary values into individual columns
    for item in nihss_items:
        graded_df[f"{item}_gold"] = graded_df["gold_scores"].apply(
            lambda x: int(x.get(item, 0)) if str(x.get(item, "NA")).isdigit() else None)
        graded_df[f"{item}_pred"] = graded_df["pred_scores"].apply(
            lambda x: int(x.get(item, 0)) if str(x.get(item, "NA")).isdigit() else None)

    # Compute NIHSS total scores (ignoring NA values)
    graded_df["NIHSS_gold_total"] = graded_df[gold_itemized_columns].sum(axis=1, skipna=True)
    graded_df["NIHSS_pred_total"] = graded_df[pred_itemized_columns].sum(axis=1, skipna=True)

    # Compute 5-class NIHSS labels
    graded_df["NIHSS_Class_gold"] = graded_df["NIHSS_gold_total"].apply(map_to_label)
    graded_df["NIHSS_Class_pred"] = graded_df["NIHSS_pred_total"].apply(map_to_label)

    # Compute evaluation metrics
    accuracy = accuracy_score(graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    kappa = cohen_kappa_score(graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])

    #use try-except to handle potential issues with wilcoxon and pearsonr
    try:
        w_stat, w_p_value = wilcoxon(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")
        print("shape of graded_df:", graded_df.shape)
        print("graded_df[NIHSS_gold_total].tolist(), graded_df[NIHSS_pred_total].tolist():", graded_df["NIHSS_gold_total"].tolist(), graded_df["NIHSS_pred_total"].tolist())
        w_stat, w_p_value = None, None

    try:
        p_stat, p_p_value = pearsonr(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    except Exception as e:
        print(f"Pearson correlation failed: {e}")
        print("shape of graded_df:", graded_df.shape)
        print("graded_df[NIHSS_gold_total].tolist(), graded_df[NIHSS_pred_total].tolist():", graded_df["NIHSS_gold_total"].tolist(), graded_df["NIHSS_pred_total"].tolist())
        p_stat, p_p_value = None, None

    #averaging pairwise QWK on 5 classes, for ordinal categorical subjects, using (graded_df["NIHSS_Class_gold"], graded_df["NIHSS_Class_pred"])
    # Define the correct order of categories
    ordinal_mapping = {"0-4": 0, "5-10": 1, "11-15": 2, "16-21": 3, ">=22": 4}

    # Convert categorical labels to ordinal integers
    graded_df["NIHSS_Class_gold_num"] = graded_df["NIHSS_Class_gold"].map(ordinal_mapping)
    graded_df["NIHSS_Class_pred_num"] = graded_df["NIHSS_Class_pred"].map(ordinal_mapping)

    qwk_score = compute_qwk(graded_df["NIHSS_Class_gold_num"], graded_df["NIHSS_Class_pred_num"])
    #ICC(2,1), for investigating correlation between raters on signle-subject continuous data, on adjudicated scores,
    # using (graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    #icc_data = pd.DataFrame({
    #    "subject": np.repeat(np.arange(len(graded_df)), 2),  # Repeat each subject twice
    #    "rater": np.tile(["gold", "pred"], len(graded_df)),
    #    "score": np.concatenate([graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"]])})
    #Saves the ICC(2,1) data
    icc_data = graded_df[['NIHSS_gold_total', 'NIHSS_pred_total']].melt(ignore_index=False).reset_index()
    print("DEBUG: csv data shape ->", output_csv)
    output_csv_icc = output_csv.replace(".csv", "_iccdata.csv")
    print("DEBUG: ICC output path ->", output_csv_icc)
    if not os.path.exists(output_csv_icc):
        # Save the ICC data to CSV if it doesn't already exist
        os.makedirs(os.path.dirname(output_csv_icc), exist_ok=True)
    icc_data.to_csv(output_csv_icc, index=False)

    try:
        ICC = compute_icc(icc_data)
    except:
        print("Error computing ICC, returning None")
        print("icc_data shape:", icc_data.shape)
        print("icc_data columns:", icc_data.columns)
        print("icc_data.head():", icc_data.head())
        ICC = None


    ICC_all = compute_icc_all(icc_data)
    #MSE, for investigating the difference between the adjudicated scores and the predicted scores,
    # using (graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])
    MSE = compute_mse(graded_df["NIHSS_gold_total"], graded_df["NIHSS_pred_total"])

    #normalized MSE
    MSE_rescaled = compute_mse(
        np.array(graded_df["NIHSS_gold_total"]) / 42,
        np.array(graded_df["NIHSS_pred_total"]) / 42
    )

    # Save detailed extracted scores and predictions to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    graded_df.to_csv(output_csv, index=False)

    # Create a structured JSON dictionary
    results_dict = \
    {
        "eval_params":
            {
#            "Sample_IDs": graded_df["Sample_ID"].tolist(),
            "data sample size": len(graded_df["Sample_ID"].tolist()),
            "base model":task.base_model,
            "temperature":task.temperature,
#            "max_tokens":task.max_tokens, #deleted, as it is not used in the evaluation
            "task": task.task,
            # "OneTimePrompting_method": task.OneTimePrompting_method
            "trial_num": task.trial_num,
            },
        "metrics":
            {
        "Accuracy": accuracy,
        "Cohen's Kappa": kappa,
        "Wilcoxon": {"W-statistic": w_stat, "p-value": w_p_value},
        "Pearson": {"Correlation": p_stat, "p-value": p_p_value},
        "QWK": qwk_score,
        "ICC": ICC,
        "ICC_all": ICC_all,
        "MSE": MSE,
        "MSE_rescaled": MSE_rescaled,
            }
    }

    # Save to a JSON file
    #json_output_path = output_csv.replace(".csv", ".json")  # Change .csv extension to .json
    # Ensure the "metrics" subfolder exists
    metrics_folder = os.path.join(os.path.dirname(output_csv), "metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    # Save JSON inside the "metrics" folder
    json_output_path = os.path.join(metrics_folder, os.path.basename(output_csv).replace(".csv", ".json"))
    print("results_dict", results_dict)

    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(results_dict, json_file, indent=4, default=convert_numpy)

    print(f" Evaluation metrics and Sample_IDs saved as JSON: {json_output_path}")

    # Return the dictionary as before
    return results_dict

###############################################################################
###############################################################################
# Helper functions for building metrics tree across runs
def _is_leaf(d: dict) -> bool:
    """Leaf-dict = all its values are *not* list/dict."""
    return isinstance(d, dict) and all(
        not isinstance(v, (list, dict)) for v in d.values()
    )

def _is_empty_value(v: Any) -> bool:
    if v is None:                               return True
    if isinstance(v, str) and v.strip() == "":  return True
    if isinstance(v, float) and math.isnan(v):  return True
    if isinstance(v, (list, dict)) and len(v) == 0: return True
    return False

def _is_empty_leaf(d: dict) -> bool:
    return all(_is_empty_value(v) for v in d.values())

# Main function to build a metrics tree across multiple runs
'''
def build_metrics_tree_across_runs(
    runs: List[Dict[str, Any]],
    metrics_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    *,
    empty_placeholder: dict | None = None,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    # gather every key that ever appears at this level
    keys = set().union(*(d.keys() for d in runs if isinstance(d, dict)))

    for key in keys:
        # collect the corresponding children for all runs (if present)
        children = [
            run[key]
            for run in runs
            if isinstance(run, dict) and key in run and run[key] is not None
        ]

        if not children:
            result[key] = {} if empty_placeholder is None else empty_placeholder
            continue

        if all(_is_leaf(c) for c in children):
            # drop fully empty leaves
            non_empty = [c for c in children if not _is_empty_leaf(c)]

            if not non_empty:  # they were all empty
                result[key] = {} if empty_placeholder is None else empty_placeholder
            else:
                df = pd.DataFrame(non_empty)
                # ensure NumPy scalars become pure Python; helps JSON dumping
                metrics = {k: (v.item() if hasattr(v, "item") else v)
                           for k, v in metrics_fn(df).items()}
                result[key] = metrics
        else:
            normalised_children = [
                c if isinstance(c, dict) else {} for c in children
            ]
            result[key] = build_metrics_tree_across_runs(
                normalised_children, metrics_fn, empty_placeholder=empty_placeholder
            )

    return result
'''
def build_metrics_tree_across_runs(
    runs: List[Dict[str, Any]],
    metrics_fn: Callable[[tuple], Dict[str, Any]], # Accepts a tuple (DataFrame, prev_concatenated_key_names)
    *,
    empty_placeholder: dict | None = None,
    _prefix: List[str] = None
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    _prefix = _prefix or []

    # Collect all keys at this level
    keys = set().union(*(d.keys() for d in runs if isinstance(d, dict)))

    for key in keys:
        current_path = _prefix + [key]

        # Collect values for this key from all runs
        children = [
            run[key]
            for run in runs
            if isinstance(run, dict) and key in run and run[key] is not None
        ]

        if not children:
            result[key] = {} if empty_placeholder is None else empty_placeholder
            continue

        if all(_is_leaf(c) for c in children):
            # Remove empty leaves
            non_empty = [c for c in children if not _is_empty_leaf(c)]

            if not non_empty:
                result[key] = {} if empty_placeholder is None else empty_placeholder
            else:
                df = pd.DataFrame(non_empty)
                concate_key_names = "__".join(current_path)

                # Pass (df, concate_key_names) as a tuple
                metrics = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in metrics_fn((df, concate_key_names)).items()
                }

                result[key] = metrics
        else:
            # Recursive step
            normalized_children = [
                c if isinstance(c, dict) else {} for c in children
            ]
            result[key] = build_metrics_tree_across_runs(
                normalized_children,
                metrics_fn,
                empty_placeholder=empty_placeholder,
                _prefix=current_path
            )

    return result
###############################################################################
###############################################################################
# The eventual version of the evaluation function for 4 splits and all ablation experiments
# Necessary auxiliary function: evaluate_performance_withGPT_InputGradedDF
def evaluate_performance_4Splits_AllAblExpr(ListOfLog_4fold, task, output_csv, IF_ReturnNone = True):
    # 1 · load logs -----------------------------------------------------------

    if not isinstance(ListOfLog_4fold, list) or len(ListOfLog_4fold) != 4: #if ListOfLog_4fold is not a list of 4 logs, but a path to one single log file
        if isinstance(ListOfLog_4fold, str):
            with open(ListOfLog_4fold, 'r', encoding='utf-8') as f:
                logs = json.load(f)

    logs = copy.deepcopy(ListOfLog_4fold)  # make a copy to avoid modifying the original list

    print("!!!length of logs sent as input in the function evaluate_performance_4Splits_AllAblExpr:", len(logs))

    # 2 · build metrics tree  -------------------------------------------------
    '''
    results_dict = build_metrics_tree_across_runs(
        logs,
        lambda x: evaluate_performance_withGPT_InputGradedDF(pd.DataFrame(x),
                                                             task,
                                                             output_csv)
    )
    '''

    results_dict = build_metrics_tree_across_runs(
        logs,
        lambda x: evaluate_performance_withGPT_InputGradedDF(x,
                                                             task,
                                                             output_csv)
    ) # x is a tuple (DataFrame, prev_concatenated_key_names)

    #print("!!! results_dict", results_dict)

    # 3 · save metrics JSON ----------------------------------------------------
    metrics_folder = os.path.join(os.path.dirname(output_csv), "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    json_output_path = os.path.join(metrics_folder,
                                    os.path.basename(output_csv).replace(".csv", ".json"))

    with open(json_output_path, "w", encoding="utf-8") as jf:
        json.dump(results_dict, jf, indent=4, default=convert_numpy)

    print(f"Evaluation metrics and Sample_IDs saved as JSON: {json_output_path}")

    if IF_ReturnNone:
        return None
    else:
        return results_dict
