"""
fix_extracted_metadata.py
Patches existing extracted/*.json files in place, without re-running the
full extraction pipeline:
  1. Fixes the pesticide catalog's crop/disease fields ("raw data" -> "all")
     directly in metadata.xlsx.
  2. Re-writes each extracted JSON's "metadata" block using the corrected
     metadata.xlsx, converting pandas NaN -> JSON null (valid JSON) instead
     of the literal NaN token.

Run this once after confirming the fixes below look right, then re-check
a file in VS Code to confirm the "Value expected" warning is gone.
"""

import os
import json
import pandas as pd

METADATA_FILE = "metadata.xlsx"
EXTRACTED_DIR = "extracted"

df = pd.read_excel(METADATA_FILE)

# --- Fix 1: pesticide catalog crop/disease values ---
pesticide_mask = df["document_type"] == "pesticide"
if (df.loc[pesticide_mask, "crop"] == "raw data").any():
    df.loc[pesticide_mask, "crop"] = "all"
    df.loc[pesticide_mask, "disease"] = "all"
    df.to_excel(METADATA_FILE, index=False)
    print("Fixed pesticide row(s): crop/disease set to 'all'. metadata.xlsx updated.")
else:
    print("Pesticide row already looks correct, no change made to metadata.xlsx.")


def clean_value(v):
    """Converts pandas NaN/NaT to None so json.dump writes valid `null`
    instead of the invalid `NaN` token."""
    if pd.isna(v):
        return None
    return v


patched_count = 0
missing_count = 0

for _, row in df.iterrows():
    file_path = row["file_path"]
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    json_path = os.path.join(EXTRACTED_DIR, base_name + ".json")

    if not os.path.exists(json_path):
        missing_count += 1
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_metadata = {k: clean_value(v) for k, v in row.to_dict().items()}
    data["metadata"] = clean_metadata

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)

    patched_count += 1

print(f"\nPatched {patched_count} extracted JSON file(s).")
if missing_count:
    print(f"{missing_count} file(s) in metadata.xlsx had no matching extracted JSON (skipped).")
print("allow_nan=False was used, so if any file still had an invalid NaN,")
print("this script would have raised an error instead of silently writing it again.")
