# src/reconcile_metadata.py

from pathlib import Path
import argparse
import re
import pandas as pd


MANUAL_CONTROL_MAPPING = {
    "ANTONIETTA P": "Antonella P",
}

EXCLUDED_PARTICIPANTS = {
    ("ELDERLY_HC", "LISCO G"): "Unresolved metadata identity",
    ("ELDERLY_HC", "PORCELLI A"): "Unresolved metadata identity",
    ("ELDERLY_HC", "SUMMO L"): "Unresolved metadata identity",
    ("ELDERLY_HC", "VITO L"): "Unresolved metadata identity",
    ("PD", "17-28/MARIO B"): "Unresolved metadata identity",
}

TARGET_SAMPLE_RATE = 16000


def normalize(value):
    """Normalize names and folder labels for cautious matching."""
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_person(value):
    """Normalize a participant folder name, including an optional range prefix."""
    value = normalize(value)
    if "/" in value:
        value = value.split("/")[-1].strip()
    return value


def find_column(df, candidates):
    """Find a column using normalized column names."""
    normalized = {
        normalize(col).replace("_", " "): col for col in df.columns
    }
    for candidate in candidates:
        key = normalize(candidate).replace("_", " ")
        if key in normalized:
            return normalized[key]
    return None


def read_metadata(path, group):
    """
    Read the spreadsheets used in this project.
    The workbooks have their column headers on the second row.
    """
    df = pd.read_excel(path, header=1)
    df = df.dropna(how="all").copy()

    name_col = find_column(df, ["name", "first name", "firstname"])
    surname_col = find_column(df, ["surname", "last name", "lastname"])
    sex_col = find_column(df, ["sex", "gender"])
    age_col = find_column(df, ["age"])

    if name_col is None or surname_col is None:
        raise ValueError(
            f"Could not find name/surname columns in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    result = pd.DataFrame({
        "metadata_group": group,
        "metadata_name": df[name_col].astype("string").str.strip(),
        "metadata_surname": df[surname_col].astype("string").str.strip(),
        "metadata_sex": df[sex_col] if sex_col else pd.NA,
        "metadata_age": pd.to_numeric(df[age_col], errors="coerce")
        if age_col else pd.NA,
    })

    result["metadata_person"] = (
        result["metadata_name"].fillna("").str.strip()
        + " "
        + result["metadata_surname"].fillna("").str.strip().str[:1]
    ).map(normalize_person)

    result = result[result["metadata_person"] != ""].copy()
    return result.reset_index(drop=True)


def build_metadata_index(metadata):
    """Return participant-name candidates without silently resolving duplicates."""
    return {
        name: group.to_dict("records")
        for name, group in metadata.groupby("metadata_person", dropna=False)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pd-xlsx", required=True)
    parser.add_argument("--hc-xlsx", required=True)
    parser.add_argument("--young-xlsx", required=True)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = pd.read_csv(manifest_path)

    required = {
        "relative_path", "group_folder", "label",
        "participant_folder", "sample_rate", "possible_task_code"
    }
    missing = required - set(recordings.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    # Read participant metadata from all three spreadsheets.
    pd_meta = read_metadata(args.pd_xlsx, "PD")
    hc_meta = read_metadata(args.hc_xlsx, "ELDERLY_HC")
    young_meta = read_metadata(args.young_xlsx, "YOUNG_HC")
    metadata = pd.concat([pd_meta, hc_meta, young_meta], ignore_index=True)

    # Convert the folder/group information into a consistent group label.
    def get_group(row):
        group_text = normalize(row["group_folder"])
        label_text = normalize(row["label"])
        combined = group_text + " " + label_text

        if "YOUNG" in combined:
            return "YOUNG_HC"
        if "ELDERLY" in combined or "HEALTHY" in combined or "CONTROL" in combined:
            return "ELDERLY_HC"
        if "PARKINSON" in combined or label_text == "PD" or group_text == "PD":
            return "PD"

        # Common dataset folder names
        if "28 PEOPLE WITH PARKINSON" in combined:
            return "PD"
        if "22 ELDERLY HEALTHY CONTROL" in combined:
            return "ELDERLY_HC"
        if "15 YOUNG HEALTHY CONTROL" in combined:
            return "YOUNG_HC"
        return "UNKNOWN"

    recordings["cohort_group"] = recordings.apply(get_group, axis=1)
    recordings["participant_name_normalized"] = (
        recordings["participant_folder"].map(normalize_person)
    )

    # Build a participant-level table first, so every recording from the same
    # person receives the same inclusion decision.
    participants = recordings[
        ["cohort_group", "participant_folder", "participant_name_normalized"]
    ].drop_duplicates().copy()

    metadata_index = build_metadata_index(metadata)
    matched_rows = []
    match_statuses = []
    include_flags = []
    exclusion_reasons = []
    manual_flags = []

    for _, person in participants.iterrows():
        group = person["cohort_group"]
        folder_name = person["participant_name_normalized"]
        folder_key = folder_name

        # Manual control mapping explicitly approved for this project.
        manual = False
        if group == "ELDERLY_HC" and folder_name in MANUAL_CONTROL_MAPPING:
            folder_key = normalize_person(MANUAL_CONTROL_MAPPING[folder_name])
            manual = True

        # The PD folder includes the range prefix, but the participant name
        # itself is the final component. Preserve the full folder separately.
        candidates = metadata_index.get(folder_key, [])
        candidate_groups = [c for c in candidates if c["metadata_group"] == group]

        status = "unmatched"
        matched = None

        if len(candidate_groups) == 1:
            matched = candidate_groups[0]
            status = "manual_match" if manual else "exact_name_match"
        elif len(candidate_groups) > 1:
            status = "ambiguous_duplicate_name"

        exclusion_key = (group, normalize(person["participant_folder"]))
        if exclusion_key not in EXCLUDED_PARTICIPANTS:
            # Also support the specified PD path if folder values contain
            # the range prefix separately.
            exclusion_key = (group, normalize(
                f'{person["participant_folder"]}'
            ))

        reason = EXCLUDED_PARTICIPANTS.get(exclusion_key)
        if reason is None and group == "PD" and folder_name == "MARIO B":
            reason = "Unresolved metadata identity"

        # Young controls are not part of the primary HC-vs-PD experiment.
        if group == "YOUNG_HC":
            reason = "Young healthy controls excluded from primary analysis"

        include = (
            group in {"ELDERLY_HC", "PD"}
            and reason is None
            and status in {"exact_name_match", "manual_match"}
        )

        if reason:
            status = "excluded_unresolved_identity" if "Unresolved" in reason else status
        elif group == "UNKNOWN":
            reason = "Could not determine cohort group"
            include = False

        matched_rows.append(matched or {})
        match_statuses.append(status)
        include_flags.append(bool(include))
        exclusion_reasons.append(reason or "")
        manual_flags.append(manual)

    participants["metadata_match_status"] = match_statuses
    participants["include_primary_cohort"] = include_flags
    participants["exclusion_reason"] = exclusion_reasons
    participants["manual_mapping_applied"] = manual_flags

    # Attach metadata fields without guessing ambiguous identities.
    for col in [
        "metadata_name", "metadata_surname", "metadata_sex",
        "metadata_age", "metadata_group"
    ]:
        participants[col] = [
            item.get(col, pd.NA) if item else pd.NA for item in matched_rows
        ]

    # Apply sample-rate eligibility at participant/recording level.
    recordings = recordings.merge(
        participants[
            [
                "cohort_group", "participant_folder",
                "metadata_match_status", "include_primary_cohort",
                "exclusion_reason", "manual_mapping_applied",
                "metadata_name", "metadata_surname",
                "metadata_sex", "metadata_age"
            ]
        ],
        on=["cohort_group", "participant_folder"],
        how="left",
    )

    recordings["sample_rate"] = pd.to_numeric(
        recordings["sample_rate"], errors="coerce"
    )
    recordings["sample_rate_eligible"] = (
        recordings["sample_rate"] == TARGET_SAMPLE_RATE
    )
    recordings["include_primary_recording"] = (
        recordings["include_primary_cohort"]
        & recordings["sample_rate_eligible"]
    )

    recordings.loc[
        ~recordings["sample_rate_eligible"]
        & recordings["exclusion_reason"].fillna("").eq(""),
        "exclusion_reason"
    ] = "Recording is not sampled at 16 kHz"

    # Save outputs.
    participants.to_csv(
        output_dir / "primary_cohort_participants.csv", index=False
    )
    recordings.to_csv(
        output_dir / "primary_cohort_manifest.csv", index=False
    )

    recordings[recordings["include_primary_recording"]].to_csv(
        output_dir / "primary_cohort_recordings_included.csv", index=False
    )

    review = participants[
        participants["metadata_match_status"].isin(
            ["unmatched", "ambiguous_duplicate_name", "excluded_unresolved_identity"]
        )
    ]
    review.to_csv(output_dir / "metadata_match_review.csv", index=False)

    # The matched table retains the manual decision and exclusions.
    participants.to_csv(
        output_dir / "participant_metadata_matched.csv", index=False
    )

    included_people = participants[participants["include_primary_cohort"]]
    group_counts = included_people["cohort_group"].value_counts().to_dict()

    summary = [
        "PRIMARY COHORT RECONCILIATION SUMMARY",
        "====================================",
        f"Participant rows: {len(participants)}",
        f"Included elderly healthy controls: {group_counts.get('ELDERLY_HC', 0)}",
        f"Included Parkinson's participants: {group_counts.get('PD', 0)}",
        f"Total included participants before sample-rate filtering: {len(included_people)}",
        f"Included recordings at {TARGET_SAMPLE_RATE} Hz: "
        f"{int(recordings['include_primary_recording'].sum())}",
        "",
        "Manual mapping: ANTONIETTA P -> Antonella P",
        "Excluded unresolved identities: LISCO G, PORCELLI A, SUMMO L, "
        "elderly-control VITO L, and PD 17-28/Mario B.",
        "Young healthy controls excluded from the primary HC-vs-PD analysis.",
        "Inspect metadata_match_review.csv before reporting demographics.",
    ]
    (output_dir / "metadata_reconciliation_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\n".join(summary))
    print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()