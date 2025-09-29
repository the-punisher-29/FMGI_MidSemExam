import subprocess
import sys

# Configuration
input_dir = "test_images"
output_dir = "output_images"
config_file = "configs/convnextB_768.yaml"
model_weights = "models/sed_model_base.pth"

# Build command
cmd = [
    sys.executable,
    "demo/demo_for_vis.py",
    "--config-file", config_file,
    "--input", f"{input_dir}/*",
    "--output", output_dir,
    "--opts",
    "MODEL.WEIGHTS", model_weights
]

print("Running demo with:")
print(f"  Input: {input_dir}")
print(f"  Output: {output_dir}")
print(f"  Model: {model_weights}")
print()

# Run
subprocess.run(cmd)