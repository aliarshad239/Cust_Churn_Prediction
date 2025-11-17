import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config
import pandas as pd

# Load configuration
cfg = load_config("config/params.yaml")

# Load dataset path from config
data_path = cfg.data["raw_path"]

# Read dataset
df = pd.read_csv(data_path)

print("✅ Dataset loaded successfully!")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nSample data:\n", df.head(3))
