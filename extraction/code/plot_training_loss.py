from pathlib import Path
import json
import matplotlib.pyplot as plt

# define paths
code_dir = Path("/projectnb/noc-lab/mrktracy/stroke_study/extraction/code")
trainer_state_path = code_dir / "LLaMA-Factory" / "outputs" / "nihss_full_lora_70b_ep30" / "trainer_state.json"
output_plot_path = code_dir / "outputs" / "training_loss_plot.png"

# Load trainer state
with open(trainer_state_path, "r", encoding="utf-8") as f:
    state = json.load(f)

# Extract training loss
steps, losses = [], []
for entry in state.get("log_history", []):
    if "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, linestyle="-", linewidth=2.5)
plt.title("Training Loss vs. Step")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot_path)

print(f"Training loss plot saved to {output_plot_path}")