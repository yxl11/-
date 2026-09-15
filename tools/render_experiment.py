"""Render the selected experiment directly from saved, separate source logs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sling_sim.experiment_visualization import plot_experiment, animate_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/experiment_v1"))
    parser.add_argument("--animation", action="store_true")
    args = parser.parse_args()
    root = args.root
    planning = dict(np.load(root/"planning/state_log.npz", allow_pickle=False))
    replay = dict(np.load(root/"replay/ideal/state_log.npz", allow_pickle=False))
    config = json.loads((root/"planning/config_snapshot.json").read_text(encoding="utf-8"))
    output = root/"media"
    output.mkdir(parents=True, exist_ok=True)
    plot_experiment(planning, replay, config, output)
    if args.animation:
        animate_experiment(planning, replay, config, output)


if __name__ == "__main__":
    main()
