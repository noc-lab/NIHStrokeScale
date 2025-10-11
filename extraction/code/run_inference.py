
import json
import csv
import time
from pathlib import Path
from statistics import mean, stdev
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# === CONFIGURATION ===
MODEL_NAME = "Llama-3.3-70B-Instruct"
TOKENIZER_PATH = "meta-llama/Llama-3.3-70B-Instruct"
PROMPTING_METHOD = "ZS"
CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
ADAPTER_DIR = CODE_DIR / "LLaMA-Factory" / "outputs" / "llama_3_3_70b_zs_qlora_ep30"
OUTPUT_DIR = CODE_DIR / "outputs" / f"{MODEL_NAME}_{PROMPTING_METHOD}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = ROOT / "data" / "llama_factory_format" / "test.json"
TRAIN_PATH = ROOT / "data" / "llama_factory_format" / "train.json"

JSONL_OUT_PATH = OUTPUT_DIR / "predictions_all.jsonl"
CSV_ALL_OUT_PATH = OUTPUT_DIR / "predictions_all.csv"
CSV_TEST_OUT_PATH = OUTPUT_DIR / "predictions_test.csv"
CSV_TRAIN_OUT_PATH = OUTPUT_DIR / "predictions_train.csv"
STATS_CSV_PATH = OUTPUT_DIR / "inference_stats_summary.csv"

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === LOAD BASE MODEL AND ADAPTER ===
config = PeftConfig.from_pretrained(str(ADAPTER_DIR))
quant_config = BitsAndBytesConfig(load_in_4bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
)

model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()

# === PIPELINE ===
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# === INFERENCE FUNCTION ===
def run_inference_on_split(examples, split_name):
    records = []
    print(f"\nRunning inference on {split_name} set...")

    for ex in tqdm(examples, desc=f"Generating {split_name} predictions"):
        prompt = ex["instruction"].strip() + "\n" + ex["input"].strip()
        start = time.perf_counter()
        outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
        end = time.perf_counter()

        prediction = outputs[0]["generated_text"].split(prompt)[-1].strip()
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(prediction))
        total_tokens = input_tokens + output_tokens
        inference_time = end - start

        record = {
            "input": ex["input"].strip(),
            "prediction": prediction,
            "reference": ex["output"].strip(),
            "split": split_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "inference_time_sec": inference_time
        }

        records.append(record)

    return records

# === LOAD FULL TEST AND TRAIN SETS ===
with open(TEST_PATH, "r", encoding="utf-8") as f_test:
    test_examples = [json.loads(line) for line in f_test]

with open(TRAIN_PATH, "r", encoding="utf-8") as f_train:
    train_examples = [json.loads(line) for line in f_train]

test_records = run_inference_on_split(test_examples, "test")
train_records = run_inference_on_split(train_examples, "train")

if "train_records" in globals():
    all_records = test_records + train_records
else:
    all_records = test_records

# === WRITE JSONL ===
with open(JSONL_OUT_PATH, "w", encoding="utf-8") as f_out:
    for rec in all_records:
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()

# === WRITE CSVs ===
def write_csv(path, data):
    with open(path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

if "test_records" in globals() and test_records:
    write_csv(CSV_TEST_OUT_PATH, test_records)
if "train_records" in globals() and train_records:
    write_csv(CSV_TRAIN_OUT_PATH, train_records)
if "all_records" in globals() and all_records:
    write_csv(CSV_ALL_OUT_PATH, all_records)

# === WRITE SUMMARY STATS ===
input_tokens_list = [r["input_tokens"] for r in all_records]
output_tokens_list = [r["output_tokens"] for r in all_records]
total_tokens_list = [r["total_tokens"] for r in all_records]
timings = [r["inference_time_sec"] for r in all_records]

summary_stats = {
    "Metric": ["input_tokens", "output_tokens", "total_tokens", "inference_time_sec"],
    "Mean": [
        mean(input_tokens_list),
        mean(output_tokens_list),
        mean(total_tokens_list),
        mean(timings)
    ],
    "Standard Deviation": [
        stdev(input_tokens_list),
        stdev(output_tokens_list),
        stdev(total_tokens_list),
        stdev(timings)
    ]
}

with open(STATS_CSV_PATH, "w", encoding="utf-8", newline="") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=["Metric", "Mean", "Standard Deviation"])
    writer.writeheader()
    for i in range(len(summary_stats["Metric"])):
        writer.writerow({
            "Metric": summary_stats["Metric"][i],
            "Mean": summary_stats["Mean"][i],
            "Standard Deviation": summary_stats["Standard Deviation"][i]
        })

print(f"\nWrote predictions to: {CSV_ALL_OUT_PATH}")
print(f"Wrote stats to: {STATS_CSV_PATH}")