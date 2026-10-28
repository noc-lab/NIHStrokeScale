# 🧠 NIH Stroke Scale (NIHSS) Inference Project

This repository provides end-to-end automation for **NIH Stroke Scale
(NIHSS)** score inference, model evaluation, and post-processing using
large language models (LLMs).\
It integrates multi-stage pipelines for **data extraction**,
**model-based inference**, and **evaluation** using reproducible,
modular code.

------------------------------------------------------------------------

## 📁 Project Structure

    NIHStrokeScale/
    ├── extraction/
    │   └── code/
    │       ├── plot_training_loss.py
    │       ├── prepare_data.py
    │       ├── radar_factory.py
    │       ├── requirements.txt
    │       ├── run_evaluation.py
    │       ├── run_inference.py
    │       └── README.pdf
    │
    ├── post_processing/
    │   └── code/
    │       ├── .ipynb_checkpoints/
    │       ├── build_finalTable.ipynb
    │       ├── build_mapfile.ipynb
    │       ├── build_matrices.ipynb
    │       ├── build_smallTable.ipynb
    │       ├── build_tidyLong.ipynb
    │       ├── convert_gptOutputs.ipynb
    │       └── README.pdf
    │
    ├── scripts/
    │
    ├── src/
    │   └── Neuro_SMEval/
    │       ├── tasks/
    │       │   └── prompts/
    │       │       ├── AggrPrompt.py
    │       │       ├── general.py
    │       │       └── task_AggrPrompt.py
    │       ├── chat_utils.py
    │       ├── ComputeAcrossFolds_utils.py
    │       ├── eval_withGPT_AllAblExpr.py
    │       ├── method.py
    │       └── __init__.py
    │
    └── requirements.txt
    └── readme.md

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the Repository

``` bash
git clone <your_repo_url>
cd NIHStrokeScale
```

### 2. Create and Activate Virtual Environment (Python 3.12)

``` bash
python3.12 -m venv venv
source venv/bin/activate
```

Alternatively, using Conda on Linux:

``` bash
conda create -n nihss_infer python=3.12 -y
conda activate nihss_infer
```

### 3. Install Dependencies

``` bash
pip install --upgrade pip
pip install -r extraction/code/requirements.txt
```

------------------------------------------------------------------------

## 🧩 Workflow Overview

This project consists of three major stages:

### 1️⃣ Data Extraction (`extraction/code/`)

Scripts to preprocess datasets, run model inference, and evaluate
performance.

  ---------------------------------------------------------------------------
  File                      Description
  ------------------------- -------------------------------------------------
  `prepare_data.py`         Loads and cleans NIHSS dataset for model input

  `run_inference.py`        Runs model-based inference on text data

  `run_evaluation.py`       Evaluates predictions using QWK, ICC, accuracy,
                            etc.

  `plot_training_loss.py`   Visualizes training and evaluation loss curves

  `radar_factory.py`        Generates radar plots for comparative
                            visualizations
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

### 2️⃣ Model Evaluation (`src/Neuro_SMEval/`)

Implements evaluation logic for multiple folds and aggregation of
results.

  -----------------------------------------------------------------------
  Module                          Purpose
  ------------------------------- ---------------------------------------
  `ComputeAcrossFolds_utils.py`   Combine metrics across cross-validation
                                  folds

  `eval_withGPT_AllAblExpr.py`    Evaluate model-generated outputs

  `chat_utils.py`                 Utilities for LLM chat inference

  `method.py`                     Experimental design and method
                                  definitions

  `tasks/prompts/`                Contains task-specific prompt templates
  -----------------------------------------------------------------------

------------------------------------------------------------------------

### 3️⃣ Post-Processing (`post_processing/code/`)

Jupyter notebooks for formatting and integrating model outputs into
final tables and charts.

  -----------------------------------------------------------------------
  Notebook                            Function
  ----------------------------------- -----------------------------------
  `build_finalTable.ipynb`            Create the final evaluation table

  `build_mapfile.ipynb`               Generate mapping between cases and
                                      outputs

  `build_matrices.ipynb`              Construct metric matrices

  `convert_gptOutputs.ipynb`          Convert raw GPT results into
                                      structured format
  -----------------------------------------------------------------------

------------------------------------------------------------------------

### 4️⃣ Running the Main Experiments (`scripts/`)

The bash script automates running large-scale model experiments (e.g.,
Llama-70B).

``` bash
chmod +x scripts/bash_run70B.sh
cd scripts
./bash_run70B.sh
```

Monitor logs:

``` bash
tail -f run_logs/run_Llama-3.3-70B-Instruct_temp0_seed42_EA1.log
```

------------------------------------------------------------------------

## 📊 Outputs

-   **Logs:** `logs_AllAblExpr/`
-   **Evaluations:** `evals_AllAblExpr/`
-   **Results (metrics):** `results_AllAblExpr/`

Each experiment produces: - JSON logs of model outputs\
- Fold-wise metrics (Accuracy, QWK, ICC)\
- Aggregated metrics across folds (CSV and visual plots)

------------------------------------------------------------------------

## 🔧 Common Troubleshooting

-   **CUDA Memory Issues** → Script auto-clears GPU memory using
    `torch.cuda.empty_cache()`\
-   **Model Not Found** → Ensure Hugging Face models are in
    `hf_dld_cache/`\
-   **Permission Denied** → Run `chmod +x scripts/bash_run70B.sh`

------------------------------------------------------------------------

## 🧑‍💻 Authors

**NIH Stroke Scale Research Team, 2025**\
Boston University / NOC Lab

------------------------------------------------------------------------

## 📜 License

This project is for **research and evaluation purposes only**, endorsed by MIT license for scientific distribution.
