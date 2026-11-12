# ItalianPVS research phase 2 — remaining analyses

Experiment: `ITALIAN-PHASE2-REMAINING-20261030`. This directory contains new sensitivity analyses, D1/D2 and second-variant signal characterization, candidate connected-speech features, a local human audio review system, dataset compatibility research, and a manuscript draft. Historical VA1 and `italian_multitask_v1` outputs were read as inputs and not overwritten.

## Run the analyses

From the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python src/italian_research_phase2.py --dataset-root "/Users/Shaan/Desktop/datasets/italian"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python src/generate_phase2_diagnostics.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python src/validate_phase2.py
```

The review-materials command is currently a library function rather than an executable module; the reproducible invocation is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from pathlib import Path; from src.italian_multitask.review_materials import create_review_materials; create_review_materials(Path(".").resolve(), Path("results/italian_research_phase2").resolve())'
```

## Launch the local review interface (macOS)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.italian_multitask.review_app --dataset-root "/Users/Shaan/Desktop/datasets/italian" --experiment-root "results/italian_research_phase2"
```

It uses Tkinter, matplotlib, Praat via Parselmouth, and the macOS `afplay` command. Navigate with Previous/Next, listen to the original WAV, inspect the waveform/spectrogram/F0/formant/RMS display, fill annotations, and save. For D1/D2, manual event onsets can be entered as comma-separated seconds and a manual syllable count recorded. Manual annotations go only to `review/manual_annotations.csv`; they never modify automated QC or source files. The queue currently has 159 distinct recordings, and all remain unresolved until a reviewer saves an annotation. The summary CSVs refresh when the app opens and after each save.

## Review boundaries

- Automated clipping/silence flags and candidate segmentation events are review prompts, not exclusions or validated labels.
- The D1/D2 candidate peaks, interval summaries, and pause fractions are not treated as syllable counts or participant-level acoustic outcomes before human segmentation review.
- Connected-speech VAD is fixed at −35 dB below each recording's 95th-percentile frame RMS for the primary descriptive table, with −30/−40 dB sensitivity. It remains an unvalidated candidate segmentation.
- Second vowel variants remain distinct codes. Their descriptive paired differences are not repeatability estimates.
- No classifier or cross-language training was performed.
