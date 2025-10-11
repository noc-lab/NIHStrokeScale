from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import json
import os
import pandas as pd
from collections import defaultdict


def compute_AvgVarStdSE_AllFolds(
    folds_or_paths: List[Union[str, Any]],
    *,
    ddof: int = 1,
    validate_structure: bool = True,
) -> Any:
    """
    Aggregate a list of equally-structured nested list/dict objects.

    Accepts either:
      1) A list of file paths (strings) to .json or .txt files containing JSON; OR
      2) A list of nested list/dict/tuple objects directly.

    For every numeric leaf across folds, compute:
      - avg (mean)
      - var (sample variance, ddof=1 by default)
      - std (sample standard deviation)
      - SE  (standard error of the mean)

    Non-numeric leaves are passed through from the first fold (assumed identical).

    If `validate_structure=True`, raises a ValueError if the items
    do NOT share the exact same nested structure (container types, dict keys, list/tuple lengths).
    """

    if not isinstance(folds_or_paths, list) or len(folds_or_paths) == 0:
        raise ValueError("Input must be a non-empty list of paths or nested objects.")

    all_str = all(isinstance(x, str) for x in folds_or_paths)
    none_str = all(not isinstance(x, str) for x in folds_or_paths)
    if not (all_str or none_str):
        raise TypeError("Do not mix file paths and in-memory objects in the same list.")

    # ---- Load JSONs if strings were passed ----
    if all_str:
        folds: List[Any] = []
        for path in folds_or_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            _, ext = os.path.splitext(path.lower())
            if ext not in {".json", ".txt"}:
                raise ValueError(f"Unsupported file type for {path}; use .json or .txt.")
            with open(path, "r", encoding="utf-8") as f:
                if ext == ".json":
                    folds.append(json.load(f))
                else:
                    folds.append(json.loads(f.read()))
    else:
        folds = folds_or_paths

    # ---- Structure validation ----
    if validate_structure:
        _assert_same_structure(folds)

    # ---- Helpers ----
    def _is_numeric(x: Any) -> bool:
        # exclude bool (a subclass of int)
        return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)

    def _to_float(x: Any) -> float:
        return np.nan if x is None else float(x)

    def _summarize_numeric(values: Sequence[Any]) -> Dict[str, float]:
        arr = np.array([_to_float(v) for v in values], dtype=float)
        n_eff = int(np.sum(~np.isnan(arr)))
        mean = float(np.nanmean(arr)) if n_eff > 0 else np.nan
        if n_eff <= ddof:
            var = std = se = np.nan
        else:
            var = float(np.nanvar(arr, ddof=ddof))
            std = float(np.sqrt(var))
            se = float(std / np.sqrt(n_eff))
        return {"avg": mean, "var": var, "std": std, "SE": se}

    def _recurse(values_at_node: List[Any]) -> Any:
        sample = values_at_node[0]

        if isinstance(sample, dict):
            return {
                k: _recurse([v[k] for v in values_at_node])
                for k in sample.keys()
            }

        if isinstance(sample, (list, tuple)):
            L = len(sample)
            out_seq = [_recurse([v[i] for v in values_at_node]) for i in range(L)]
            return type(sample)(out_seq)

        if _is_numeric(sample):
            return _summarize_numeric(values_at_node)

        # Non-numeric leaf: pass through (identical across folds by assumption)
        return sample

    return _recurse(folds)


def compress_quadruples(obj):
    if isinstance(obj, dict):
        # Check if it's a 4-key stats dict
        if set(obj.keys()) == {"avg", "var", "std", "SE"}:
            var_value = obj.get("var")
            if var_value == 0 or var_value == 0.0:
                return obj["avg"]
            else:
                return obj  # Don't compress if var ≠ 0

        # Recurse deeper
        return {k: compress_quadruples(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [compress_quadruples(v) for v in obj]

    else:
        return obj


# a useful function but deprecated (not used in the codebase)
def tables_to_string_dfs(tables: dict) -> dict[str, pd.DataFrame]:
    """
    Convert each table (dict of dicts) into a pandas DataFrame
    where all row names, column names, and cell values are strings.
    """
    string_dfs = {}
    for metric_name, tbl in tables.items():
        df = pd.DataFrame(tbl)

        # Ensure indices (rows) are strings
        df.index = df.index.map(str)
        # Ensure columns are strings
        df.columns = df.columns.map(str)
        # Convert all contents to strings
        df = df.astype(str)

        string_dfs[metric_name] = df
    return string_dfs


def _assert_same_structure(items: List[Any]) -> None:
    """
    Raise ValueError if `items` do not share the same nested structure.
    Structure equality means:
      - All corresponding nodes are of the same container type (dict/list/tuple/leaf)
      - Dicts have identical key sets
      - Lists/Tuples have identical lengths
      - Leaves can be any (numeric or not); we only care about container structure
    """
    if len(items) < 2:
        return

    def _kind(x: Any) -> str:
        if isinstance(x, dict):
            return "dict"
        if isinstance(x, list):
            return "list"
        if isinstance(x, tuple):
            return "tuple"
        return "leaf"

    def _walk(group: List[Any], path: str) -> None:
        first = group[0]
        k = _kind(first)

        # All kinds must match
        for i, g in enumerate(group[1:], start=1):
            kg = _kind(g)
            if kg != k:
                raise ValueError(
                    f"Structure mismatch at {path}: expected '{k}', found '{kg}' in item #{i}."
                )

        if k == "dict":
            first_keys = set(first.keys())
            for i, g in enumerate(group[1:], start=1):
                if set(g.keys()) != first_keys:
                    missing = first_keys - set(g.keys())
                    extra = set(g.keys()) - first_keys
                    raise ValueError(
                        f"Dict keys mismatch at {path}. "
                        f"Missing in item #{i}: {sorted(missing)}; "
                        f"Extra in item #{i}: {sorted(extra)}"
                    )
            # Recurse per key
            for key in first.keys():
                _walk([x[key] for x in group], f"{path}.{key}")

        elif k in {"list", "tuple"}:
            L = len(first)
            for i, g in enumerate(group[1:], start=1):
                if len(g) != L:
                    raise ValueError(
                        f"Sequence length mismatch at {path}: expected {L}, found {len(g)} in item #{i}."
                    )
            # Recurse per index
            for idx in range(L):
                _walk([x[idx] for x in group], f"{path}[{idx}]")

        else:
            # leaf: nothing else to check for structure
            return

    _walk(items, "root")


def flatten_nested_dict(input_data: Union[dict, list],
                        metrics: List[str] = ["avg", "var", "std", "SE"]
                        ) -> pd.DataFrame:
    rows = []

    def recurse(curr, key_path):
        if isinstance(curr, dict):
            # Check if it's a metrics dict
            if all(m in curr for m in metrics):
                row = {"concate_name": "__".join(key_path)}
                for m in metrics:
                    row[m] = curr.get(m)
                rows.append(row)
                return
            # Otherwise go deeper
            for k, v in curr.items():
                recurse(v, key_path + [str(k)])

        elif isinstance(curr, list):
            for i, item in enumerate(curr):
                recurse(item, key_path + [f"[{i}]"])

    if isinstance(input_data, list):
        for i, entry in enumerate(input_data):
            recurse(entry, [f"[{i}]"])
    else:
        recurse(input_data, [])

    df = pd.DataFrame(rows)

    #print("shape of the DataFrame after flattening:", df.shape)
    #print("columns in the DataFrame:", df.columns)

    df = df[["concate_name"] + metrics]
    return df


def generate_metric_tables_From5Coldf(df,
                                      metrics_to_save,
                                      error_type='std',
                                      output_folder='../metric_tables',
                                      PlusMinusSign = "+-"):
    """
    Generates and saves tables for each specified metric suffix from the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with columns ['concate_name', 'avg', 'std', 'SE'].
        metrics_to_save (list[str]): List of metric suffixes to extract.
        error_type (str): One of 'std' or 'SE'. Default is 'std'.
        output_folder (str): Directory where the output CSVs will be saved.

    Returns:
        None
    """
    assert error_type in ['std', 'SE'], "error_type must be either 'std' or 'SE'."
    os.makedirs(output_folder, exist_ok=True)

    for metric in metrics_to_save:
        # Filter rows that end with the metric suffix
        df_filtered = df[df['concate_name'].str.lower().str.endswith(metric.lower())].copy() # Case-insensitive match
        if df_filtered.empty:
            print(f"Skipping '{metric}': no matching rows found.")
            continue

        # Extract row name (first token) and the column name (middle parts)
        row_names = df_filtered['concate_name'].str.split('__').str[0]
        # Extract everything after the first '__' and remove the matched suffix
        col_names = df_filtered['concate_name'].apply(
            lambda x: '__'.join(x.split('__')[1:]).removesuffix(f"__{metric}").removesuffix(metric)
        )

        df_filtered['row_name'] = row_names
        df_filtered['col_name'] = col_names

        # Compose formatted value
        df_filtered['value'] = df_filtered['avg'].round(6).astype(str) + PlusMinusSign + df_filtered[error_type].round(
            6).astype(str)

        # Handle duplicates
        pivot_table = df_filtered.groupby(['row_name', 'col_name'])['value'].first().unstack(fill_value='')

        # Save to CSV
        filename = metric.replace(" ", "_").replace("/", "_") + ".csv"
        pivot_table.to_csv(os.path.join(output_folder, filename))

        print(f"Saved table for '{metric}' to '{filename}'")


if __name__ == '__main__':
    print("Hello from ComputeAcrossFolds_utils.py!")