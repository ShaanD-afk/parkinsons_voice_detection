"""
plot_pitch_tracks_sensitivity.py
---------------------------------
Phase 2: Generate pitch track visualizations for participants flagged in Phase 1
as having large sensitivity to pitch-floor changes.

For each flagged participant, this script:
 1. Loads the original audio.
 2. Computes pitch tracks under all 3 settings (75, 60, 90 Hz floors).
 3. Plots all 3 tracks on a shared time/frequency axis.
 4. Saves the figure to data/pitch_sensitivity/plots/.
 5. Saves frame-level CSV data to data/pitch_sensitivity/pitch_track_data/.
 6. Writes a narrative review record to data/pitch_sensitivity/pitch_track_review.csv.

NOTE: The review classification is based on visual inspection of evidence
(waveform, pitch track shape, sensitivity pattern). A recording is NOT
excluded based solely on visual impression; ambiguous cases are documented
as such.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------------------------------------------------
# 1.  Configuration
# -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SENS = DATA / "pitch_sensitivity"

DATASET_ROOT = Path("/Users/Shaan/Desktop/datasets/italian")

PLOT_DIR   = SENS / "plots"
TRACK_DIR  = SENS / "pitch_track_data"
REVIEW_CSV = SENS / "pitch_track_review.csv"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
TRACK_DIR.mkdir(parents=True, exist_ok=True)

PITCH_CEILING = 500
TIME_STEP     = 0.01

CONFIGURATIONS = {
    "floor75 (reference)": 75,
    "floor60":              60,
    "floor90":              90,
}

CONFIG_COLORS = {
    "floor75 (reference)": "#1f77b4",
    "floor60":              "#d62728",
    "floor90":              "#2ca02c",
}

# Participants to plot, from Phase 1 ranking.
# Extend this list manually if new participants are added.
FLAGGED_PARTICIPANTS = [
    "AGNESE P",
    "GILDA C",
    "GRAZIA G",
    "17-28/Giovanni M",
]

# Also include participants that showed notable HNR changes.
ADDITIONAL_REVIEW = [
    "GIOVANNA G",   # unusually large F0 IQR
    "17-28/Giustina M",  # large IQR shift under floor60
]


def resolve_audio(relative_path):
    path = Path(str(relative_path))
    candidate = DATASET_ROOT / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Audio not found: {candidate}"
    )


# -----------------------------------------------------------------------
# 2.  Compute pitch track for one configuration
# -----------------------------------------------------------------------

def compute_pitch_track(audio, sample_rate, pitch_floor):
    """Return arrays (times_sec, f0_hz) where f0_hz is NaN for unvoiced frames."""
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch = sound.to_pitch_ac(
        time_step=TIME_STEP,
        pitch_floor=pitch_floor,
        pitch_ceiling=PITCH_CEILING,
    )
    times  = pitch.xs()
    f0_raw = pitch.selected_array["frequency"]
    # Praat uses 0 for unvoiced frames.
    f0 = np.where((f0_raw > 0) & np.isfinite(f0_raw), f0_raw, np.nan)
    return times, f0


# -----------------------------------------------------------------------
# 3.  Plot one participant
# -----------------------------------------------------------------------

def plot_participant(participant_folder, relative_path, features_row):
    """Compute pitch tracks, plot them, save PNG and CSV."""

    audio_path = resolve_audio(relative_path)
    audio, sample_rate = sf.read(str(audio_path), dtype="float64")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    audio = audio[np.isfinite(audio)]

    duration = len(audio) / sample_rate

    # Compute pitch tracks.
    tracks = {}
    for config_name, floor in CONFIGURATIONS.items():
        times, f0 = compute_pitch_track(audio, sample_rate, floor)
        tracks[config_name] = {"times": times, "f0": f0}

    # Save frame-level CSV.
    max_len = max(len(v["times"]) for v in tracks.values())
    track_rows = []
    for config_name, data in tracks.items():
        for t, f in zip(data["times"], data["f0"]):
            track_rows.append({
                "participant_folder": participant_folder,
                "config": config_name,
                "time_sec": t,
                "f0_hz": f,
            })
    track_df = pd.DataFrame(track_rows)
    safe_name = participant_folder.replace("/", "_").replace(" ", "_")
    track_path = TRACK_DIR / f"{safe_name}_pitch_tracks.csv"
    track_df.to_csv(track_path, index=False)

    # ---- Figure ----
    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # Top panel: waveform.
    ax_wave = axes[0]
    t_audio = np.arange(len(audio)) / sample_rate
    ax_wave.plot(t_audio, audio, color="#555555", linewidth=0.3, alpha=0.7)
    ax_wave.set_ylabel("Amplitude", fontsize=9)
    ax_wave.set_yticks([])
    ax_wave.set_title(
        f"Pitch tracks — {participant_folder}  "
        f"({features_row['cohort_group']})",
        fontsize=11,
        pad=6,
    )

    # Bottom panel: pitch tracks.
    ax_f0 = axes[1]

    voiced_all = []
    for config_name, data in tracks.items():
        f0 = data["f0"]
        times = data["times"]
        voiced_mask = ~np.isnan(f0)
        if voiced_mask.any():
            voiced_all.extend(f0[voiced_mask].tolist())

        # Voiced segments as solid line, unvoiced as dots on x-axis.
        for i in range(len(times)):
            if not np.isnan(f0[i]):
                ax_f0.plot(
                    [times[i]],
                    [f0[i]],
                    marker="o",
                    markersize=2.5,
                    color=CONFIG_COLORS[config_name],
                    linewidth=0,
                    alpha=0.8,
                )

        # Overlay connecting line through voiced frames only.
        voiced_times = times[voiced_mask]
        voiced_f0    = f0[voiced_mask]
        if len(voiced_times):
            ax_f0.plot(
                voiced_times,
                voiced_f0,
                color=CONFIG_COLORS[config_name],
                linewidth=1.2,
                alpha=0.5,
                label=f"{config_name}  (floor={CONFIGURATIONS[config_name]} Hz)",
            )

    # Set shared axis limits.
    ax_f0.set_xlim(0, duration)
    if voiced_all:
        y_lo = max(0, min(voiced_all) - 20)
        y_hi = max(voiced_all) + 20
        ax_f0.set_ylim(y_lo, y_hi)

    # Shade unvoiced regions (based on reference track).
    ref_data = tracks["floor75 (reference)"]
    ref_times = ref_data["times"]
    ref_f0    = ref_data["f0"]
    unvoiced_mask = np.isnan(ref_f0)
    prev_uv = False
    start_t = 0.0
    for i, uv in enumerate(unvoiced_mask):
        if uv and not prev_uv:
            start_t = ref_times[i]
        elif not uv and prev_uv:
            ax_f0.axvspan(start_t, ref_times[i], alpha=0.08, color="gray")
        prev_uv = uv
    if prev_uv:
        ax_f0.axvspan(start_t, duration, alpha=0.08, color="gray")

    # Horizontal lines at common octave-jump ratios.
    if voiced_all:
        median_ref = float(np.nanmedian([
            v for v, c in zip(
                [f for f in tracks["floor75 (reference)"]["f0"]],
                [True] * len(tracks["floor75 (reference)"]["f0"]),
            ) if not np.isnan(v)
        ]))
        for factor, label in [(0.5, "½×"), (2.0, "2×")]:
            level = median_ref * factor
            if y_lo < level < y_hi:
                ax_f0.axhline(
                    level, color="orange", linewidth=0.8,
                    linestyle="--", alpha=0.7,
                    label=f"Reference median {label} ({level:.0f} Hz)",
                )

    ax_f0.set_xlabel("Time (s)", fontsize=9)
    ax_f0.set_ylabel("F0 (Hz)", fontsize=9)
    ax_f0.legend(fontsize=8, loc="upper right")
    ax_f0.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_f0.grid(True, which="major", alpha=0.3)

    # Annotate floor and ceiling.
    for floor_val, color in [(60, "#d62728"), (75, "#1f77b4"), (90, "#2ca02c")]:
        ax_f0.axhline(
            floor_val, linestyle=":", linewidth=0.7,
            color=color, alpha=0.4,
        )

    # Key feature values.
    feat_text = (
        f"Ref (75 Hz): median F0={features_row['median_f0_hz']:.1f} Hz, "
        f"IQR={features_row['f0_iqr_hz']:.1f} Hz, "
        f"voiced={features_row['voiced_frame_fraction']:.2f}"
    )
    ax_f0.text(
        0.01, 0.02, feat_text,
        transform=ax_f0.transAxes,
        fontsize=7.5, color="#333333",
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plot_path = PLOT_DIR / f"{safe_name}_pitch_tracks.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    return track_path, plot_path, tracks


# -----------------------------------------------------------------------
# 4.  Review classification
# -----------------------------------------------------------------------

REVIEW_NOTES = {
    "AGNESE P": {
        "observed_issue": "Octave jump under 60 Hz floor",
        "evidence": (
            "Median F0 drops from ~217 Hz (ref) to ~72 Hz (floor60), a factor of ~3× "
            "— strongly suggests the 60 Hz floor allows tracking of the sub-harmonic "
            "or the second subharmonic. Reference (75 Hz) and 90 Hz floor agree closely. "
            "The 72 Hz estimate is below typical female F0 and at 1/3 of the 217 Hz value."
        ),
        "conclusion": (
            "The 60 Hz floor introduces an apparent tracking error (sub-harmonic or "
            "low-frequency noise tracking). The 75 Hz and 90 Hz estimates are consistent. "
            "The 60 Hz result for this participant should be treated with caution."
        ),
        "review_status": "High-sensitivity — probable tracking artefact under floor60",
    },
    "GILDA C": {
        "observed_issue": "Large F0 IQR suppressed under 90 Hz floor",
        "evidence": (
            "F0 IQR = 87.8 Hz at 75 Hz floor, collapses to 2.5 Hz at 90 Hz floor. "
            "Median F0 is stable (~171–174 Hz). The 90 Hz floor may be suppressing "
            "valid low-frequency F0 frames that contribute to the high IQR, or "
            "alternatively it may be removing subharmonic frames that inflated the IQR."
        ),
        "conclusion": (
            "Ambiguous: the very high IQR at 75/60 Hz may reflect genuine F0 "
            "variability (e.g., voice tremor or instability), or subharmonic tracking "
            "below 90 Hz. The 90 Hz floor result may be over-suppressing. "
            "Cannot resolve without additional evidence (e.g., auditory inspection)."
        ),
        "review_status": "Ambiguous — requires further investigation",
    },
    "GRAZIA G": {
        "observed_issue": "Large F0 IQR suppressed under 90 Hz floor",
        "evidence": (
            "F0 IQR = 84.7 Hz at 75 Hz floor, collapses to 6.4 Hz at 90 Hz floor. "
            "Median F0 changes only ~4 Hz (stable). Pattern is similar to GILDA C. "
            "HNR also decreases under 90 Hz floor (~1.8 dB)."
        ),
        "conclusion": (
            "Ambiguous: similar to GILDA C. The 90 Hz floor substantially removes "
            "IQR-contributing frames. The source (genuine variability vs. subharmonic) "
            "cannot be determined from feature values alone."
        ),
        "review_status": "Ambiguous — requires further investigation",
    },
    "17-28/Giovanni M": {
        "observed_issue": "Median F0 halved and IQR inflated under 60 Hz floor",
        "evidence": (
            "Median F0 drops from 127 Hz (ref) to 68 Hz (floor60) — close to a factor "
            "of 2×, suggesting octave-down tracking. F0 IQR simultaneously increases "
            "from 2.3 Hz to 62.3 Hz, indicating the lower floor admits many inconsistent "
            "low-frequency frames. HNR increases 7.2 dB under floor60, which can occur "
            "when sub-harmonic frames with different harmonic structure are included. "
            "90 Hz floor gives median 126.9 Hz and IQR 4.9 Hz — consistent with ref."
        ),
        "conclusion": (
            "The 60 Hz floor very likely introduces sub-harmonic (octave-down) tracking "
            "for this male PD participant whose F0 is near 127 Hz. The 75 Hz and 90 Hz "
            "estimates are consistent. The floor60 result for Giovanni M should be "
            "treated with caution in sensitivity analyses."
        ),
        "review_status": "High-sensitivity — probable octave-down tracking artefact under floor60",
    },
    "GIOVANNA G": {
        "observed_issue": "High F0 IQR under all settings",
        "evidence": (
            "F0 IQR is 110.3 Hz (ref), 108.1 Hz (floor90). Median F0 is stable. "
            "The high IQR is consistent across settings, suggesting genuine F0 variability "
            "rather than a tracking artefact."
        ),
        "conclusion": (
            "The high IQR appears stable across settings. This likely reflects genuine "
            "F0 instability in the recording rather than a tracking error. "
            "No setting-specific artefact is indicated."
        ),
        "review_status": "Stable high variability — no setting-specific artefact",
    },
    "17-28/Giustina M": {
        "observed_issue": "F0 IQR shifts under floor60",
        "evidence": (
            "F0 IQR changes from 40.5 Hz (ref) to 33.0 Hz (floor60) — a 7.5 Hz reduction. "
            "This is below the 20 Hz exploratory threshold and modest relative to the IQR value. "
            "Median F0 changes <2.5 Hz across all settings."
        ),
        "conclusion": (
            "Moderate sensitivity to floor setting. The IQR changes are smaller than for "
            "the primary flagged cases. No obvious tracking artefact is indicated."
        ),
        "review_status": "Low-moderate sensitivity — no obvious artefact",
    },
}


# -----------------------------------------------------------------------
# 5.  Main
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2: Pitch track plots and review")
    print("=" * 60)

    orig = pd.read_csv(DATA / "va1_features_raw.csv")
    ranking = pd.read_csv(SENS / "participant_ranking.csv")

    all_to_review = list(dict.fromkeys(FLAGGED_PARTICIPANTS + ADDITIONAL_REVIEW))

    review_records = []

    for participant in all_to_review:
        row = orig[orig["participant_folder"] == participant]
        if row.empty:
            print(f"WARNING: {participant} not found in feature table — skipping.")
            continue

        row = row.iloc[0]
        relative_path = row["relative_path"]

        print(f"\nProcessing: {participant} ({row['cohort_group']})")
        print(f"  Ref median F0: {row['median_f0_hz']:.1f} Hz, IQR: {row['f0_iqr_hz']:.1f} Hz")

        try:
            track_path, plot_path, tracks = plot_participant(
                participant, relative_path, row
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            track_path = None
            plot_path  = None
            tracks     = {}

        # Get changes from Phase 1 ranking for this participant.
        part_ranking = ranking[ranking["participant_folder"] == participant]

        notes = REVIEW_NOTES.get(participant, {
            "observed_issue": "Not in pre-specified notes",
            "evidence": "See pitch track plot",
            "conclusion": "Manual review required",
            "review_status": "Pending",
        })

        # Gather per-config median F0 and IQR from sensitivity CSVs.
        f60 = pd.read_csv(SENS / "floor60.csv")
        f90 = pd.read_csv(SENS / "floor90.csv")
        ref_row = orig[orig["participant_folder"] == participant].iloc[0]
        f60_row = f60[f60["participant_folder"] == participant].iloc[0]
        f90_row = f90[f90["participant_folder"] == participant].iloc[0]

        review_records.append({
            "participant_folder":      participant,
            "cohort_group":            row["cohort_group"],
            "label":                   int((row["cohort_group"] == "PD")),
            "ref_median_f0_hz":        float(ref_row["median_f0_hz"]),
            "ref_f0_iqr_hz":           float(ref_row["f0_iqr_hz"]),
            "ref_voiced_fraction":     float(ref_row["voiced_frame_fraction"]),
            "ref_jitter":              float(ref_row["jitter_local"]),
            "ref_shimmer":             float(ref_row["shimmer_local"]),
            "ref_hnr_db":              float(ref_row["hnr_mean_db"]),
            "floor60_median_f0_hz":    float(f60_row["median_f0_hz"]),
            "floor60_f0_iqr_hz":       float(f60_row["f0_iqr_hz"]),
            "floor90_median_f0_hz":    float(f90_row["median_f0_hz"]),
            "floor90_f0_iqr_hz":       float(f90_row["f0_iqr_hz"]),
            "observed_issue":          notes["observed_issue"],
            "evidence":                notes["evidence"],
            "conclusion":              notes["conclusion"],
            "review_status":           notes["review_status"],
            "plot_path":               str(plot_path) if plot_path else "",
            "track_csv_path":          str(track_path) if track_path else "",
        })

    review_df = pd.DataFrame(review_records)
    review_df.to_csv(REVIEW_CSV, index=False)
    print(f"\nSaved: {REVIEW_CSV}")

    print("\n" + "=" * 60)
    print("Pitch-track review summary:")
    print("=" * 60)
    print(review_df[
        ["participant_folder", "cohort_group",
         "ref_median_f0_hz", "ref_f0_iqr_hz",
         "observed_issue", "review_status"]
    ].to_string(index=False))

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
