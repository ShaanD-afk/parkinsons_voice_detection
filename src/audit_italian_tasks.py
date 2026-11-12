
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFESTS = [
    PROJECT_ROOT / "data" / "italian_manifest.csv",
    PROJECT_ROOT / "data" / "primary_cohort_manifest.csv",
]

manifest_path = next((p for p in MANIFESTS if p.exists()), None)

if manifest_path is None:
    raise FileNotFoundError(
        "Neither italian_manifest.csv nor "
        "primary_cohort_manifest.csv was found."
    )

df = pd.read_csv(manifest_path)

print("\n" + "=" * 70)
print("ITALIANPVS DATASET AUDIT")
print("=" * 70)

print(f"\nManifest: {manifest_path}")
print(f"Total rows: {len(df)}")

print("\nAVAILABLE COLUMNS")
print(df.columns.tolist())

print("\nFIRST FIVE ROWS")
print(df.head().to_string(index=False))

# Identify likely columns.
path_candidates = [
    c for c in df.columns
    if any(x in c.lower() for x in ["path", "file", "audio"])
]

group_candidates = [
    c for c in df.columns
    if any(x in c.lower() for x in ["group", "diagnosis", "label", "class"])
]

participant_candidates = [
    c for c in df.columns
    if any(x in c.lower() for x in ["participant", "subject", "speaker", "person", "id"])
]

sample_rate_candidates = [
    c for c in df.columns
    if any(x in c.lower() for x in ["sample_rate", "samplerate", "sr"])
]

print("\nLIKELY PATH COLUMNS:", path_candidates)
print("LIKELY GROUP COLUMNS:", group_candidates)
print("LIKELY PARTICIPANT COLUMNS:", participant_candidates)
print("LIKELY SAMPLE-RATE COLUMNS:", sample_rate_candidates)

# Print categorical distributions.
print("\nCATEGORICAL COLUMN DISTRIBUTIONS")

for col in df.columns:
    if df[col].dtype == "object":
        unique_count = df[col].nunique(dropna=False)

        if unique_count <= 60:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).to_string())

# Inspect folder/path structures.
for col in path_candidates:
    print(f"\nPATH STRUCTURE: {col}")

    paths = df[col].dropna().astype(str)

    for path in paths.head(15):
        print(" ", path)

    # Extract the first folder component.
    first_folder = paths.apply(
        lambda x: Path(x).parts[0] if Path(x).parts else "UNKNOWN"
    )

    print("\nFirst folder component counts:")
    print(first_folder.value_counts().to_string())

# Sampling rate distribution.
for col in sample_rate_candidates:
    print(f"\nSAMPLE RATE DISTRIBUTION: {col}")
    print(df[col].value_counts(dropna=False).to_string())

print("\n" + "=" * 70)
print("AUDIT COMPLETE — NO FILES MODIFIED")
print("=" * 70)
