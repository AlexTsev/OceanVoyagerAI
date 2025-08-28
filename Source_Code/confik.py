import pandas as pd
import json

# Load dataset
df = pd.read_csv("../Dataset/vesseldataset.csv")

# Select numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

config_dict = {}

for col in numeric_cols:
    avg = df[col].mean()
    std = df[col].std()
    # normalize column names for config keys (replace spaces, special chars if needed)
    key_avg = f"{col}_avg"
    key_std = f"{col}_std"
    config_dict[key_avg] = float(avg)
    config_dict[key_std] = float(std)

# Save as a Python file
with open("./myconfik.py", "w") as f:
    f.write("myconfik = {\n")
    for k, v in config_dict.items():
        f.write(f"    '{k}': {v},\n")
    f.write("}\n")

print("✅ myconfik_generated.py created with mean/std for all numeric columns.")
