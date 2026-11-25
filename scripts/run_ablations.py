from resnet import runner
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


def main():
    ''' 
    TODO
    bottleneck architecture for res block
    multiple process training
    validation set
    '''
    if len(sys.argv) < 2:
        sys.exit(1)
    ablation_idx = int(sys.argv[1])

    configs = []
    with open(CONFIGS / "ablations.json", "r") as f:
        configs = json.load(f)
    config = configs[ablation_idx]

    for seed in range(5):
        runner.run(RESULTS, seed,
                   config["layer"], config["n"], config["res"])


if __name__ == "__main__":
    main()
