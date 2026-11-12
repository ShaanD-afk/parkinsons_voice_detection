from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description="Run the ItalianPVS multi-task QC and first-vowel analysis work package")
    parser.add_argument("--dataset-root",required=True,help="Path to the original ItalianPVS audio directory")
    parser.add_argument("--force",action="store_true",help="Recompute cached QC and primary feature rows")
    args=parser.parse_args()
    run_pipeline(project_root,Path(args.dataset_root).expanduser().resolve(),force=args.force)


if __name__=="__main__":
    main()
