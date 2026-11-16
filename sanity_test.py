"""Root wrapper for evaluation/sanity_test.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.sanity_test import main, test_sanity

if __name__ == "__main__":
    main()
