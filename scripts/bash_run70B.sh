#!/bin/bash
# ===============================================================
# NIHSS Inference - Batch Runner
# ---------------------------------------------------------------
# Runs multiple inference experiments with varying model, seed,
# and temperature configurations using nohup background processes.
# ===============================================================

# Define model names (add more if needed)
models=(
  "meta-llama/Llama-3.3-70B-Instruct"
)

# Define temperature values
temps=(0)

# Define random seed values
seeds=(42)

# Whether to use abbreviation-expanded notes (1 = Yes, 0 = No)
If_use_notes_expand_abbrev=1

# Define log directory
run_log_dir="run_logs"
mkdir -p "$run_log_dir"  # Ensure log directory exists

# Path to main experiment script
main_script="scripts/run_withGPT_GoldSilver_AllAblExpr_250806fix_250819update.py"

# Path to Hugging Face cache (edit as needed)
hf_cache_path="../.cache"

# Begin running experiments
for model in "${models[@]}"; do
  for temp in "${temps[@]}"; do
    for seed in "${seeds[@]}"; do

      # Get model basename for log naming
      model_name=$(basename "$model")
      log_file="${run_log_dir}/run_${model_name}_temp${temp}_seed${seed}_EA${If_use_notes_expand_abbrev}.log"

      echo "🚀 Starting experiment:"
      echo "   Model: $model"
      echo "   Temp: $temp"
      echo "   Seed: $seed"
      echo "   Log:  $log_file"
      echo "----------------------------------------"

      nohup python3 "$main_script" \
        --base_model "$model" \
        --temperature "$temp" \
        --rd_seed "$seed" \
        --hf_home_cache_path "$hf_cache_path" \
        --IF_use_notes_expand_abbrev "$If_use_notes_expand_abbrev" \
        --usage_infer 1 \
        --usage_eval 0 \
        --usage_results 0 \
        > "$log_file" 2>&1 &

    done
  done
done

echo "✅ All experiments started in the background with nohup!"
echo "🪵 Check logs in the '$run_log_dir' directory."
