from resnet import runner
from pathlib import Path
import json


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
    configs = {}
    with open(CONFIGS / "ablations.json", "r") as f:
        configs = json.load(f)

    for config in configs:
        for seed in range(5):
            runner.run(RESULTS, seed, config["layer"], config["n"])


if __name__ == "__main__":
    main()
