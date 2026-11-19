"""Create count-based flat Spot inputs, preserving every original H5AD."""
from pathlib import Path
import argparse
import json
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "data_factory" / "lib")]
from mignet_ce.io.experiment_spec import load_experiment
from species_adapter import prepare_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    print(json.dumps(prepare_inputs(load_experiment(a.experiment_config), a.data_root,
                                   resume=a.resume, dry_run=a.dry_run), indent=2))


if __name__ == "__main__":
    main()
