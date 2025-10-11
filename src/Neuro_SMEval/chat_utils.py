import copy
import importlib
import pandas as pd
import numpy as np
import random
import os
import subprocess

from Neuro_SMEval.models import get_response_method


task_parameter_mapping = {
    "AggrPrompt": "AggrPrompt",
    "AggrPrompt_250805_v7": "AggrPrompt_250805_v7",
}


def import_prompts_by_task(task_name):
    parameter = task_parameter_mapping.get(task_name)
    module_name = f"Neuro_SMEval.tasks.prompts.{parameter}"

    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        raise ImportError(f"Failed to import module: {module_name}")


def select_score_RankInterval_representatives(df, N, score_col="score", seed=None):
    """
    Selects N representative rows from a DataFrame based on score intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a column of scores.
    N : int
        Number of representative rows to select.
    score_col : str, default="score"
        Name of the column in `df` to use for sorting and ranking.
    seed : int or None, default=None
        Random seed for reproducibility. If None, randomness is uncontrolled.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of N rows, each representing a chunk of the sorted score distribution.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if len(df) <= N:
        return df.sample(n=len(df), random_state=seed).reset_index(drop=True)

    # Sort by score
    df_sorted = df.sort_values(by=score_col).reset_index(drop=True)

    # Chunk sorted DataFrame into N nearly-equal parts
    chunks = np.array_split(df_sorted, N)

    selected_rows = []
    used_indices = set()

    for chunk in chunks:

        available_chunk = chunk[~chunk.index.isin(used_indices)]

        if not available_chunk.empty:
            #random choice from the available chunk
            #selected_row = available_chunk.sample(n=1, random_state=seed)

            #pick the median row from the available chunk based on score_col
            median_index = len(available_chunk) // 2
            selected_row = available_chunk.iloc[median_index:median_index + 1]

            selected_rows.append(selected_row)
            used_indices.update(selected_row.index)

        else: # If all rows in the chunk are already used, we need to select from the entire DataFrame
            # Fallback: choose any row not yet used. This case will not occur if the DataFrame's num of rows is larger than N.
            fallback_pool = df_sorted[~df_sorted.index.isin(used_indices)]
            if not fallback_pool.empty:
                selected_row = fallback_pool.sample(n=1, random_state=seed)
                selected_rows.append(selected_row)
                used_indices.update(selected_row.index)

    result_df = pd.concat(selected_rows, ignore_index=True)
    return result_df


# Convert chat messages to a prompt string manually (simple format)
def format_chat_LlamaTemplate2Text(messages):
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"<|system|>\n{content}\n"
        elif role == "user":
            formatted += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}\n"
    return formatted


def select_gpus(threshold_mb=100, force_all=1):
    '''
    Select GPUs based on their memory usage.
    If force_all is set to 1, all free GPUs will be selected.

    Args:
        threshold_mb (int): Memory threshold in MB to consider a GPU as free.
        force_all (int): If set to 1, all free GPUs will be selected without user input.
    Returns:
        None: Sets the environment variable CUDA_VISIBLE_DEVICES based on user selection.
    Usage:
        select_gpus(threshold_mb=120, force_all=1) for selecting GPUs with at least 100 MB free memory.
        select_gpus() for default behavior with threshold 100 MB and force_all set to 1.
    '''

    def get_free_gpus():
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        free = []
        print("GPU status:")
        for line in out.strip().split('\n'):
            idx, name, used = line.strip().split(', ')
            used = int(used)
            status = "FREE" if used < threshold_mb else "IN USE"
            print(f"GPU {idx}: {name}, Used: {used} MiB  {status}")
            if used < threshold_mb:
                free.append(idx)
        return free

    free_gpus = get_free_gpus()

    if force_all:
        selected = free_gpus
    else:
        choice = input("\nChoose mode - (1) Use all free GPUs, (2) Specify GPUs manually: ")
        if choice == '1':
            selected = free_gpus
        elif choice == '2':
            selected = input("Enter comma-separated GPU IDs (e.g., 0,2): ").split(',')
        else:
            print("Invalid choice"); exit()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected)
    print(f"\nos.environ['CUDA_VISIBLE_DEVICES'] = '{os.environ['CUDA_VISIBLE_DEVICES']}'")

