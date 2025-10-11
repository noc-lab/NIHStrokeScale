import pandas as pd
import json
import pathlib
import textwrap
from sklearn.model_selection import train_test_split

# Define paths
ROOT = pathlib.Path(__file__).resolve().parents[1]  # stroke_study/extraction
IN_CSV = ROOT / "data" / "NIHComplete_finalized_GoldStandardExtraction_AbbrevExpanded_250305.csv"
OUT_DIR = ROOT / "data" / "llama_factory_format"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = OUT_DIR / "train.json"
TEST_JSON = OUT_DIR / "test.json"
DATASET_INFO_JSON = OUT_DIR / "dataset_info.json"

# Column names
COL_NOTE = "Note"
COL_GOLD = "Submitted_0224"

# LLaMA Factory expects instruction-tuning format
instruction = "Extract ONLY the neurological exam from the note below."

df = pd.read_csv(IN_CSV)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

def convert_to_llama_factory_format(df, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "instruction": instruction,
                "input": str(row[COL_NOTE]).strip(),
                "output": str(row[COL_GOLD]).strip()
            }
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# Write train and test JSON files
convert_to_llama_factory_format(train_df, TRAIN_JSON)
convert_to_llama_factory_format(test_df, TEST_JSON)

# Write dataset_info.json with required "file_name" fields and alpaca formatting
config = {
    "neuro_notes": {
        "file_name": "train.json",
        "formatting": "alpaca",
        "type": "sft",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    },
    "neuro_notes_test": {
        "file_name": "test.json",
        "formatting": "alpaca",
        "type": "eval",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
}

with DATASET_INFO_JSON.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"Wrote {len(train_df):,} training examples to {TRAIN_JSON}")
print(f"Wrote {len(test_df):,} test examples to {TEST_JSON}")
print(f"Saved dataset config to {DATASET_INFO_JSON}")
