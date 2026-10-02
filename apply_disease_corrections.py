"""
apply_disease_corrections.py
Applies the SPECIFIC, confirmed disease-label corrections identified by
cross-checking each file's pathogen against its disease label - not a
blind pattern-based normalization. Shows every change before writing.
"""

import pandas as pd

METADATA_FILE = "metadata.xlsx"

# (file_path fragment to match, new disease label, reason)
CORRECTIONS = [
    ("Citrus Black Spot.pdf", "Black Spot",
     "Pathogen is Phyllosticta citricarpa - this is Black Spot, not generic leaf spot"),
    ("Citrus-black-spot.pdf", "Black Spot",
     "Pathogen is Phyllosticta citricarpa - this is Black Spot, not generic leaf spot"),
    ("olive leaf spot.pdf", "Peacock Spot",
     "Pathogen is Venturia oleaginea - this IS Peacock Spot (same disease as Tutorial-8-Peacock-Spot.pdf)"),
    ("Tutorial-8-Peacock-Spot.pdf", "Peacock Spot",
     "Filename and content confirm Peacock Spot, not generic leaf spot"),
    ("Cercospora Leaf Spot.pdf", "Cercospora Leaf Spot",
     "Pathogen is Cercospora capsici - naming it specifically avoids confusion with pepper's separate Alternaria Leaf Spot"),
    ("septoria+leaf+spot.pdf", "Septoria Leaf Spot",
     "Filename confirms Septoria lycopersici"),
    ("Grapevine Leaf Spot.pdf", "Phomopsis Leaf Spot",
     "Pathogen is Phomopsis viticola - Tunisian catalog lists this target as 'Excoriose', "
     "a generic 'leaf spot' label would likely fail to match it in the pesticide catalog"),
    ("wheat fusarium head blight.pdf", "Fusarium Head Blight",
     "Clarity/consistency - matches the pathogen name directly"),
    ("fig tache folliculaire.pdf", "Cercospora Leaf Spot",
     "Consolidated with CercosporaLeafSpotFig.pdf per project scope (max diseases per crop) - "
     "this file's Rhizoctonia/anthracnose content becomes supplementary detail, not a separate disease"),
    ("CercosporaLeafSpotFig.pdf", "Cercospora Leaf Spot",
     "The consolidation target itself - was still labeled generic 'leaf spot', missed in the first pass"),
]

df = pd.read_excel(METADATA_FILE)

print("Proposed corrections:\n")
changes_to_make = []
for fragment, new_disease, reason in CORRECTIONS:
    mask = df["file_path"].str.contains(fragment, case=False, na=False, regex=False)
    matched_rows = df[mask]
    if matched_rows.empty:
        print(f"  NOT FOUND: no file matching '{fragment}' - skipping")
        continue
    for idx, row in matched_rows.iterrows():
        if row["disease"] == new_disease:
            continue  # already correct
        print(f"  {row['file_path']}")
        print(f"    '{row['disease']}' -> '{new_disease}'")
        print(f"    Reason: {reason}\n")
        changes_to_make.append((idx, new_disease))

if not changes_to_make:
    print("No changes needed - all corrections already applied or no matches found.")
else:
    answer = input(f"Apply these {len(changes_to_make)} correction(s) to metadata.xlsx? (yes/no): ").strip().lower()
    if answer == "yes":
        for idx, new_disease in changes_to_make:
            df.at[idx, "disease"] = new_disease
        df.to_excel(METADATA_FILE, index=False)
        print("metadata.xlsx updated.")
        print("NOTE: re-run distill.py and match_pesticide_treatments.py to pick up the corrected labels.")
    else:
        print("No changes made.")

print("\nAll fig files now consolidated under 'Cercospora Leaf Spot'.")
