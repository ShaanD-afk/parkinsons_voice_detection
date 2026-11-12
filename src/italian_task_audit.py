"""Reproducible ItalianPVS task inventory and participant coverage audit.

This reads source manifests/workbooks and audio filenames only. It does not
change cohort decisions or write into data/ or the baseline result locations.
Run from any directory with: python -m src.italian_task_audit
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/Users/Shaan/Desktop/datasets/italian")
OUT = ROOT / "reports" / "italian_task_audit"
GROUP_ORDER = ["YOUNG_HC", "ELDERLY_HC", "PD"]
GROUP_FILES = {
    "YOUNG_HC": ("15 Young Healthy Control", "15 YHC.xlsx"),
    "ELDERLY_HC": ("22 Elderly Healthy Control", "Tab 3.xlsx"),
    "PD": ("28 People with Parkinson's disease", "TAB 5.xlsx"),
}
CODEBOOK_PATHS = [
    DATASET_ROOT / folder / "FILE CODES.xlsx" for folder, _ in GROUP_FILES.values()
]
EXPECTED_GROUP_RECORDINGS = {"PD": 437, "ELDERLY_HC": 349, "YOUNG_HC": 45}
EXPECTED_GROUP_PARTICIPANTS = {"PD": 28, "ELDERLY_HC": 22, "YOUNG_HC": 15}
EXPECTED_PRIMARY_PARTICIPANTS = {"ELDERLY_HC": 17, "PD": 19}
EXPECTED_PRIMARY_16K_RECORDINGS = 455
TASK_RE = re.compile(r"^([A-Z]+\d*)", re.IGNORECASE)


def norm_name(value: object) -> str:
    """Conservative display-name key used only for workbook cross-checks."""
    if pd.isna(value):
        return ""
    value = str(value).strip().upper().replace("’", "'")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("'", "")
    value = re.sub(r"[^A-Z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def participant_display(folder: str) -> str:
    return str(folder).strip().split("/")[-1]


def read_codebook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    rows = []
    for _, row in raw.iterrows():
        values = [x for x in row.tolist() if pd.notna(x)]
        if not values:
            continue
        code = next((str(v).strip().upper() for v in values if re.fullmatch(r"[A-Z]+\d+", str(v).strip(), re.I)), None)
        if code:
            description = next((str(v).strip() for v in values if str(v).strip().upper() != code), "")
            rows.append({"task_code": code, "description": description})
    return pd.DataFrame(rows).drop_duplicates().sort_values("task_code").reset_index(drop=True)


def read_workbook_roster(path: Path, group: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=1).dropna(how="all").copy()
    columns = {str(c).strip().lower(): c for c in raw.columns}
    name_col = columns.get("name")
    surname_col = columns.get("surname")
    if name_col is None or surname_col is None:
        raise ValueError(f"Name/surname columns not found in {path}: {list(raw.columns)}")
    roster = raw[raw[name_col].notna() & raw[surname_col].notna()].copy()
    roster["workbook_name"] = roster[name_col].astype(str).str.strip()
    roster["workbook_surname"] = roster[surname_col].astype(str).str.strip().str.rstrip("*")
    roster["group"] = group
    roster["identity_key"] = roster.apply(
        lambda r: norm_name(f"{r['workbook_name']} {r['workbook_surname'][:1]}"), axis=1
    )
    roster["source_workbook"] = path.name
    return roster[["group", "workbook_name", "workbook_surname", "identity_key", "source_workbook"]]


def summarize(manifest: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        subset = manifest[manifest["cohort_group"].eq(group)]
        for task in sorted(manifest["possible_task_code"].dropna().astype(str).unique()):
            part = subset[subset["possible_task_code"].astype("string").eq(task)]
            rows.append({
                "population": scenario,
                "group": group,
                "task_code": task,
                "recordings": len(part),
                "unique_participants": part["participant_id"].nunique(),
                "recordings_16000_hz": int(part["sample_rate_num"].eq(16000).sum()),
                "recordings_44100_hz": int(part["sample_rate_num"].eq(44100).sum()),
            })
    return pd.DataFrame(rows)


def make_matrix(manifest: pd.DataFrame, scenario: str, tasks: list[str]) -> pd.DataFrame:
    people = manifest[["cohort_group", "participant_folder", "participant_id"]].drop_duplicates()
    counts = manifest.pivot_table(
        index=["cohort_group", "participant_folder", "participant_id"],
        columns="possible_task_code", values="relative_path", aggfunc="count", fill_value=0,
    )
    counts = counts.reindex(columns=tasks, fill_value=0).reset_index()
    counts.columns.name = None
    counts.insert(0, "population", scenario)
    # Presence flags are binary; the separate summaries retain recording counts.
    counts[tasks] = counts[tasks].gt(0).astype(int)
    return counts


def overlaps(manifest: pd.DataFrame, scenario: str, tasks: list[str]) -> pd.DataFrame:
    rows = []
    scopes = [("ALL", manifest)] + [(g, manifest[manifest.cohort_group.eq(g)]) for g in GROUP_ORDER]
    for group, frame in scopes:
        sets = {
            task: set(frame.loc[frame.possible_task_code.eq(task), "participant_id"])
            for task in tasks
        }
        for left, right in itertools.combinations(tasks, 2):
            rows.append({
                "population": scenario, "group": group,
                "task_a": left, "task_b": right,
                "participants_task_a": len(sets[left]),
                "participants_task_b": len(sets[right]),
                "participants_with_both": len(sets[left] & sets[right]),
                "participants_only_a": len(sets[left] - sets[right]),
                "participants_only_b": len(sets[right] - sets[left]),
                "participants_with_either": len(sets[left] | sets[right]),
            })
    return pd.DataFrame(rows)


def check_row(name: str, passed: bool, observed: object, expected: object, detail: str = "") -> dict:
    return {"check": name, "status": "PASS" if passed else "FAIL", "observed": observed, "expected": expected, "detail": detail}


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate package."""
    if frame.empty:
        return "_No rows._"
    columns = [str(c) for c in frame.columns]
    values = [[str(v).replace("|", "\\|").replace("\n", " ") for v in row]
              for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines)


def main() -> int:
    full_path = ROOT / "data" / "italian_manifest.csv"
    reconciled_path = ROOT / "data" / "primary_cohort_manifest.csv"
    full = pd.read_csv(full_path, dtype={"possible_task_code": "string"})
    primary = pd.read_csv(reconciled_path, dtype={"possible_task_code": "string"})
    required = {"relative_path", "group_folder", "label", "participant_folder", "filename", "possible_task_code", "sample_rate"}
    for path, frame in [(full_path, full), (reconciled_path, primary)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing required fields: {sorted(missing)}")

    full["cohort_group"] = full["label"].astype(str)
    primary["cohort_group"] = primary["cohort_group"].astype(str)
    for frame in (full, primary):
        frame["participant_id"] = frame["cohort_group"] + "::" + frame["participant_folder"].astype(str)
        frame["sample_rate_num"] = pd.to_numeric(frame["sample_rate"], errors="coerce")
        frame["possible_task_code"] = frame["possible_task_code"].str.strip().str.upper()

    if not full["label"].equals(full["cohort_group"]):
        raise ValueError("Full manifest label/group normalization failed")
    primary_all = primary[primary["include_primary_cohort"].fillna(False).astype(bool)].copy()
    if "include_primary_recording" in primary:
        primary_16k = primary[primary["include_primary_recording"].fillna(False).astype(bool)].copy()
    else:
        primary_16k = primary_all[primary_all.sample_rate_num.eq(16000)].copy()
    primary_all = primary_all[primary_all.cohort_group.isin(["ELDERLY_HC", "PD"])]
    primary_16k = primary_16k[primary_16k.cohort_group.isin(["ELDERLY_HC", "PD"])]

    scenarios = {
        "full_dataset": full,
        "reconciled_hc_pd": primary_all,
        "reconciled_hc_pd_16khz": primary_16k,
    }
    tasks = sorted(full.possible_task_code.dropna().astype(str).unique())
    OUT.mkdir(parents=True, exist_ok=True)

    # Group/task summaries and participant presence matrices for each defined population.
    summary = pd.concat([summarize(frame, name) for name, frame in scenarios.items()], ignore_index=True)
    summary.to_csv(OUT / "task_recording_participant_summary.csv", index=False)
    for name, frame in scenarios.items():
        make_matrix(frame, name, tasks).to_csv(OUT / f"participant_task_presence_{name}.csv", index=False)
    pd.concat([overlaps(frame, name, tasks) for name, frame in scenarios.items()], ignore_index=True).to_csv(
        OUT / "pairwise_task_overlap.csv", index=False
    )

    # Binary task absences are observed coverage gaps, not presumed protocol failures.
    absence_rows = []
    for scenario, frame in scenarios.items():
        matrix = make_matrix(frame, scenario, tasks)
        for _, row in matrix.iterrows():
            for task in tasks:
                if int(row[task]) == 0:
                    absence_rows.append({"population": scenario, "group": row.cohort_group,
                                         "participant_folder": row.participant_folder,
                                         "task_code_absent_from_manifest": task,
                                         "interpretation": "Coverage absence; protocol expectation not established"})
    pd.DataFrame(absence_rows, columns=["population", "group", "participant_folder", "task_code_absent_from_manifest", "interpretation"]).to_csv(
        OUT / "participant_task_coverage_gaps.csv", index=False
    )

    # Read and compare every supplied task-code workbook and participant roster.
    codebooks = []
    roster_parts = []
    for group, (folder, roster_name) in GROUP_FILES.items():
        code_path = DATASET_ROOT / folder / "FILE CODES.xlsx"
        roster_path = DATASET_ROOT / folder / roster_name
        codes = read_codebook(code_path)
        codes["group"] = group
        codes["source_workbook"] = str(code_path)
        codebooks.append(codes)
        roster_parts.append(read_workbook_roster(roster_path, group))
    codebook_rows = pd.concat(codebooks, ignore_index=True)
    codebook_rows.to_csv(OUT / "workbook_task_code_definitions.csv", index=False)
    roster = pd.concat(roster_parts, ignore_index=True)

    observed_codes = set(tasks)
    documented_codes = set(codebook_rows.task_code)
    code_status = []
    for task in sorted(observed_codes | documented_codes):
        descs = sorted({d for d in codebook_rows.loc[codebook_rows.task_code.eq(task), "description"] if d})
        code_status.append({
            "task_code": task,
            "observed_in_manifest": task in observed_codes,
            "documented_in_codebook": task in documented_codes,
            "description_as_written": " | ".join(descs),
            "status": ("observed_and_documented" if task in observed_codes and task in documented_codes
                       else "observed_undocumented" if task in observed_codes
                       else "documented_not_observed"),
        })
    code_audit = pd.DataFrame(code_status)
    code_audit.to_csv(OUT / "task_code_reconciliation.csv", index=False)

    # Existing reconciled participant decisions are consumed as supplied, never recomputed.
    participants_path = ROOT / "data" / "primary_cohort_participants.csv"
    participant_decisions = pd.read_csv(participants_path)
    decision_cols = [c for c in ["cohort_group", "participant_folder", "participant_name_normalized",
                                  "metadata_match_status", "include_primary_cohort", "exclusion_reason",
                                  "manual_mapping_applied", "metadata_name", "metadata_surname", "metadata_age"]
                     if c in participant_decisions]
    identity = participant_decisions[decision_cols].drop_duplicates().copy()
    identity["workbook_crosscheck_key"] = identity.participant_folder.map(participant_display).map(norm_name)
    roster["workbook_crosscheck_key"] = roster.identity_key
    checks_identity = []
    for _, p in identity.iterrows():
        candidates = roster[(roster.group == p.cohort_group) & (roster.workbook_crosscheck_key == p.workbook_crosscheck_key)]
        if p.participant_name_normalized == "ANTONIETTA P":
            candidates = roster[(roster.group == p.cohort_group) & (roster.identity_key == norm_name("Antonella P"))]
        status = "matched" if len(candidates) == 1 else "ambiguous_workbook_match" if len(candidates) > 1 else "not_found_in_roster"
        checks_identity.append({**p.to_dict(), "workbook_match_count": len(candidates), "workbook_crosscheck_status": status,
                                "matched_workbook_identity": " | ".join(candidates.identity_key.tolist())})
    identity_crosscheck = pd.DataFrame(checks_identity)
    identity_crosscheck.to_csv(OUT / "participant_identity_reconciliation.csv", index=False)

    # Distinguish reused folder IDs across groups from repeated display names.
    people = full[["cohort_group", "group_folder", "participant_folder", "participant_id"]].drop_duplicates()
    people["normalized_terminal_name"] = people.participant_folder.map(participant_display).map(norm_name)
    folder_group_counts = people.groupby("participant_folder").cohort_group.nunique()
    reused_folder_names = set(folder_group_counts[folder_group_counts.gt(1)].index)
    exact_folder_collisions = people[people.participant_folder.isin(reused_folder_names)].copy()
    exact_folder_collisions["collision_type"] = "same participant_folder string in multiple groups; inspect"
    name_counts = people.groupby("normalized_terminal_name").participant_id.nunique()
    repeated_names = set(name_counts[name_counts.gt(1)].index)
    repeated_name_rows = people[people.normalized_terminal_name.isin(repeated_names)].copy()
    repeated_name_rows["collision_type"] = "same terminal name; separate group/folder identity retained"
    pd.concat([exact_folder_collisions, repeated_name_rows], ignore_index=True).drop_duplicates().to_csv(
        OUT / "participant_identity_collisions.csv", index=False
    )

    # Workbook people absent from corresponding group-folder audio inventory.
    inventory_keys = set()
    for group, frame in full.groupby("cohort_group"):
        inventory_keys |= {(group, norm_name(participant_display(x))) for x in frame.participant_folder.unique()}
    missing_roster = roster[["group", "workbook_name", "workbook_surname", "identity_key", "source_workbook"]].copy()
    missing_roster["found_by_name_key_in_audio"] = missing_roster.apply(lambda r: (r.group, r.identity_key) in inventory_keys, axis=1)
    missing_roster.to_csv(OUT / "workbook_roster_audio_crosscheck.csv", index=False)

    # Manifest/audio inventory comparison: path completeness, duplicate path and exact filename prefix checks.
    manifest_paths = full.relative_path.astype(str)
    dataset_audio = {str(p.relative_to(DATASET_ROOT)) for p in DATASET_ROOT.rglob("*")
                     if p.is_file() and p.suffix.lower() in {".wav", ".wave"}}
    path_counts = manifest_paths.value_counts()
    duplicate_paths = path_counts[path_counts.gt(1)].rename_axis("relative_path").reset_index(name="manifest_rows")
    duplicate_paths.to_csv(OUT / "duplicate_manifest_entries.csv", index=False)
    missing_audio = sorted(set(manifest_paths) - dataset_audio)
    extra_audio = sorted(dataset_audio - set(manifest_paths))
    pd.DataFrame([{"issue": "manifest_path_missing_on_disk", "relative_path": p} for p in missing_audio]
                 + [{"issue": "audio_file_absent_from_manifest", "relative_path": p} for p in extra_audio],
                 columns=["issue", "relative_path"]).to_csv(OUT / "audio_manifest_file_discrepancies.csv", index=False)

    parsed = full.filename.fillna("").astype(str).str.extract(r"^([A-Z]+\d*)", flags=re.IGNORECASE)[0].str.upper()
    name_anomalies = full.loc[parsed.fillna("").ne(full.possible_task_code.fillna("")),
                              ["group_folder", "participant_folder", "filename", "possible_task_code", "relative_path"]].copy()
    name_anomalies["parsed_filename_prefix"] = parsed.loc[name_anomalies.index]
    named_special = full[full.possible_task_code.isin(["PR1", "PR11"])][
        ["group_folder", "participant_folder", "filename", "possible_task_code", "relative_path"]
    ].copy()
    named_special["anomaly_note"] = named_special.possible_task_code.map(
        {"PR1": "Code absent from provided FILE CODES workbook; no meaning assigned",
         "PR11": "Single observed code; absent from codebook; filename prefix matches possible_task_code"})
    name_anomalies.to_csv(OUT / "filename_task_code_mismatches.csv", index=False)
    named_special.to_csv(OUT / "pr1_pr11_review.csv", index=False)

    # Reconciliation file consistency and structural validations.
    checks = []
    checks.append(check_row("Full manifest recording total", len(full) == 831, len(full), 831))
    checks.append(check_row("Full manifest unique participants", full.participant_id.nunique() == 65, full.participant_id.nunique(), 65))
    unique_people = full[["cohort_group", "participant_folder"]].drop_duplicates()
    key_duplicates = int(unique_people.duplicated().sum())
    checks.append(check_row("Participant identity keys unique by group and folder",
                            key_duplicates == 0, key_duplicates, 0))
    checks.append(check_row("Participant folder IDs are not reused across groups",
                            exact_folder_collisions.empty, len(exact_folder_collisions), 0,
                            "Repeated terminal names are separately reported and are not treated as duplicate IDs."))
    actual_groups = full.groupby("cohort_group").size().to_dict()
    checks.append(check_row("Full manifest group recording totals", actual_groups == EXPECTED_GROUP_RECORDINGS, actual_groups, EXPECTED_GROUP_RECORDINGS))
    group_people = full.groupby("cohort_group").participant_id.nunique().to_dict()
    checks.append(check_row("Full manifest group participant totals", group_people == EXPECTED_GROUP_PARTICIPANTS, group_people, EXPECTED_GROUP_PARTICIPANTS))
    primary_people = primary_all.groupby("cohort_group").participant_id.nunique().to_dict()
    checks.append(check_row("Reconciled cohort participant counts", primary_people == EXPECTED_PRIMARY_PARTICIPANTS, primary_people, EXPECTED_PRIMARY_PARTICIPANTS))
    checks.append(check_row("Reconciled 16 kHz recording total", len(primary_16k) == EXPECTED_PRIMARY_16K_RECORDINGS, len(primary_16k), EXPECTED_PRIMARY_16K_RECORDINGS))
    checks.append(check_row("Manifest relative paths unique", duplicate_paths.empty, int(duplicate_paths.manifest_rows.sum()) if len(duplicate_paths) else 0, 0))
    checks.append(check_row("Manifest paths match audio files on disk", not missing_audio and not extra_audio,
                            {"missing_on_disk": len(missing_audio), "unlisted_audio": len(extra_audio)}, {"missing_on_disk": 0, "unlisted_audio": 0}))
    checks.append(check_row("Task-code prefix consistency", name_anomalies.empty, len(name_anomalies), 0))
    checks.append(check_row("Primary cohort manifest row inventory matches full manifest",
                            set(primary.relative_path) == set(full.relative_path), len(set(primary.relative_path) ^ set(full.relative_path)), 0))
    flags = primary.drop_duplicates("participant_id").set_index("participant_id").include_primary_cohort.fillna(False).astype(bool)
    dec = participant_decisions.drop_duplicates(["cohort_group", "participant_folder"]).copy()
    dec["participant_id"] = dec.cohort_group.astype(str) + "::" + dec.participant_folder.astype(str)
    decision_flags = dec.set_index("participant_id").include_primary_cohort.fillna(False).astype(bool)
    flag_mismatch = int((flags.reindex(decision_flags.index).fillna(False) != decision_flags).sum())
    checks.append(check_row("Primary manifest decisions match participant decision table", flag_mismatch == 0, flag_mismatch, 0))
    codebook_sets = [set(x.task_code) for x in codebooks]
    checks.append(check_row("Three FILE CODES workbooks agree", all(s == codebook_sets[0] for s in codebook_sets[1:]),
                            [sorted(s) for s in codebook_sets], "same task-code set"))
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(OUT / "validation_checks.csv", index=False)

    # Population/group totals for transparent report and validation context.
    population_rows = []
    for name, frame in scenarios.items():
        for group in GROUP_ORDER:
            sub = frame[frame.cohort_group.eq(group)]
            population_rows.append({"population": name, "group": group, "recordings": len(sub),
                                    "unique_participants": sub.participant_id.nunique(),
                                    "sample_rates": ";".join(f"{int(k)} Hz: {int(v)}" for k, v in sub.sample_rate_num.value_counts().sort_index().items())})
    population = pd.DataFrame(population_rows)
    population.to_csv(OUT / "population_totals.csv", index=False)

    # Report findings without manufacturing expected task schedules or definitions.
    task_counts = summary[summary.population.eq("full_dataset")].groupby("task_code").agg(
        recordings=("recordings", "sum"), participants=("unique_participants", "sum")).reset_index()
    documented_absent = sorted(documented_codes - observed_codes)
    observed_undoc = sorted(observed_codes - documented_codes)
    n_crosscheck = identity_crosscheck.workbook_crosscheck_status.value_counts().to_dict()
    n_missing_roster = int((~missing_roster.found_by_name_key_in_audio).sum())
    roster_missing_names = missing_roster.loc[~missing_roster.found_by_name_key_in_audio].apply(
        lambda r: f"{r.group}: {r.workbook_name} {r.workbook_surname}", axis=1
    ).tolist()
    reconciliation_unmatched = identity_crosscheck.loc[
        identity_crosscheck.workbook_crosscheck_status.eq("not_found_in_roster"), "participant_folder"
    ].astype(str).tolist()
    n_gaps = len(absence_rows)
    failed = check_frame[check_frame.status.eq("FAIL")]
    status = "VALIDATION PASSED" if failed.empty else "VALIDATION FAILED — audit requires correction"
    lines = [
        "# ItalianPVS task inventory and participant coverage audit", "",
        f"**Status: {status}.** This is a manifest and workbook coverage audit; it does not extract features, test group differences, or train models.", "",
        "## Inputs and method", "",
        "Primary task field: `possible_task_code` from `data/italian_manifest.csv`. The full and reconciled manifests were compared by relative path. Participant matrices use a group-qualified participant key to prevent same-named people in different groups from being merged. `FILE CODES.xlsx` is read separately for all three groups; roster cross-checks are descriptive and do not recalculate reconciliation decisions. Audio files are checked against the supplied ItalianPVS root.", "",
        "## Population totals", "",
        markdown_table(population), "",
        "## Observed task codes", "",
        markdown_table(task_counts), "",
        f"Observed codes: {', '.join(tasks)}. Full data contain {len(full)} manifest rows, {full.participant_id.nunique()} group-qualified participants, and {len(dataset_audio)} audio files.", "",
        "## Task-code documentation findings", "",
        f"- `FB2` is documented in all three codebooks as reading phonemically balanced words, but is absent from the manifest and filename-derived task-code inventory. This audit cannot determine whether the task was not recorded, omitted from the supplied dataset, or represented under another code.",
        f"- Observed but undocumented codes: {', '.join(observed_undoc) if observed_undoc else 'none'}. `PR1` and `PR11` match their filename prefixes; the codebooks do not define either. `PR11` occurs {int(full.possible_task_code.eq('PR11').sum())} time. No meaning is assigned.",
        "- `VA1`, `VE1`, `VI1`, `VO1`, and `VU1` codebook descriptions say `2 phonation of the vocal ...`; the literal wording is retained in the codebook output and not interpreted.",
        "- Documentation codes absent from observed files: " + (", ".join(documented_absent) if documented_absent else "none") + ".", "",
        "## Files, names, and coverage gaps", "",
        f"- Manifest paths missing on disk: {len(missing_audio)}; audio files absent from manifest: {len(extra_audio)}.",
        f"- Exact duplicate manifest paths: {len(duplicate_paths)}; filename/task-code prefix mismatches: {len(name_anomalies)}.",
        f"- Workbook roster names not found by normalized name key in the corresponding group folder: {n_missing_roster}. See `workbook_roster_audio_crosscheck.csv`; this conservative comparison may flag spelling/alias differences and does not override reconciliation decisions.",
        "  - Names: " + ("; ".join(roster_missing_names) if roster_missing_names else "none") + ".",
        "- Existing participant folders not found in the corresponding workbook roster: "
        + ("; ".join(reconciliation_unmatched) if reconciliation_unmatched else "none")
        + ". The known `17-28/Mario B` exclusion does not match the workbook roster entry `Mario M`; preserve the existing exclusion pending identity evidence.",
        f"- Binary participant/task coverage gaps across the three defined populations: {n_gaps}. These are absences from the manifest only. Since the supplied materials do not establish an expected task schedule per participant, they cannot be labeled missing required recordings.",
        f"- Existing reconciliation cross-check statuses: {n_crosscheck}. See `participant_identity_reconciliation.csv` for participant-level detail; no cohort decision was changed.",
        "- Identity key uses group plus the original participant folder. Similar terminal names in different groups or numbered PD folders are not automatically merged.", "",
        "## Pairwise coverage", "",
        "`pairwise_task_overlap.csv` reports participant counts for each task pair in each population and group, including common participants and participants exclusive to either task. The three `participant_task_presence_*.csv` files are binary matrices; recording counts remain available in `task_recording_participant_summary.csv`.", "",
        "## Automated validation", "",
        markdown_table(check_frame), "",
        f"**Passed:** {(check_frame.status == 'PASS').sum()} / {len(check_frame)}. **Failed:** {len(failed)}.", "",
        "## Unresolved questions", "",
        "1. Why is documented FB2 absent from the observed recording set? Consult dataset custodians/protocol records; do not relabel another task without evidence.",
        "2. What do PR1 and PR11 represent, and is PR11 a filename anomaly or a distinct undocumented code? Retain both literal codes pending source documentation.",
        "3. Which task absences are genuinely missing recordings? A participant-level expected task schedule is not provided in the checked materials.",
        "4. Several workbook-to-folder identity comparisons can be ambiguous or unmatched; the pre-existing reconciliation decisions remain authoritative and require targeted review only if a future analysis depends on those participants.", "",
        "## Output files", "",
        "- `population_totals.csv` — recording and participant totals for each population/group.",
        "- `task_recording_participant_summary.csv` — recording and unique-participant counts by task and group for all populations.",
        "- `participant_task_presence_full_dataset.csv`, `participant_task_presence_reconciled_hc_pd.csv`, `participant_task_presence_reconciled_hc_pd_16khz.csv` — participant-by-task presence matrices.",
        "- `pairwise_task_overlap.csv` — pairwise overlap/common participants by population/group.",
        "- `participant_task_coverage_gaps.csv` — observed participant/task absences, not assertions of protocol omissions.",
        "- `workbook_task_code_definitions.csv`, `task_code_reconciliation.csv` — codebook definitions and observed/documented status.",
        "- `participant_identity_reconciliation.csv`, `workbook_roster_audio_crosscheck.csv` — identity/reconciliation cross-checks.",
        "- `participant_identity_collisions.csv` — reused folder IDs and repeated terminal names, clearly distinguished.",
        "- `duplicate_manifest_entries.csv`, `audio_manifest_file_discrepancies.csv`, `filename_task_code_mismatches.csv`, `pr1_pr11_review.csv` — integrity and anomaly detail.",
        "- `validation_checks.csv` — machine-readable validation outcomes.", "",
        "## Scope conclusion", "",
        ("All implemented structural validation checks passed. The inventory is reproducible for the supplied manifest, cohort decisions, workbooks, and audio tree. Research questions about code meaning, FB2 absence, and expected participant task schedules remain unresolved; these do not permit inference about undocumented task meanings or missing protocol-required recordings." if failed.empty else "At least one validation check failed. Do not treat this audit as complete until the failed check is understood and resolved."),
    ]
    (OUT / "task_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{status}: {len(check_frame) - len(failed)}/{len(check_frame)} checks passed; outputs: {OUT}")
    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
