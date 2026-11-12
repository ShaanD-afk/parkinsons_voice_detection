from pathlib import Path

EXPERIMENT_ID = "ITALIAN-MULTITASK-V1-20261030"
TASKS_QC = [
    "VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2",
    "D1", "D2", "B1", "B2", "FB1",
]
FIRST_VOWELS = ["VA1", "VE1", "VI1", "VO1", "VU1"]
GROUPS_PRIMARY = ["ELDERLY_HC", "PD"]
GROUPS_DESCRIPTIVE = ["YOUNG_HC", "ELDERLY_HC", "PD"]
PITCH = {"method": "Praat autocorrelation", "time_step_s": 0.01, "floor_hz": 75.0, "ceiling_hz": 500.0}
PITCH_SENSITIVITY_FLOORS = [60.0, 75.0, 90.0]
FRAME = {"frame_ms": 25.0, "hop_ms": 10.0}
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 5000
RESAMPLE_RATE = 16000
VOWEL_PHYSICAL_RANGES = {
    "f1_hz": (150.0, 1400.0),
    "f2_hz": (350.0, 4000.0),
    "f3_hz": (900.0, 5000.0),
}


def output_root(project_root: Path) -> Path:
    return project_root / "results" / "italian_multitask_v1"
