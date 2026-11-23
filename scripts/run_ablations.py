from resnet import runner
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main():
    for seed in range(1, 3):
        runner.run(RESULTS, seed)


if __name__ == "__main__":
    main()
