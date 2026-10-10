import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backbone  # noqa: E402
from common.cli import evaluate_main  # noqa: E402

if __name__ == "__main__":
    evaluate_main(backbone)
