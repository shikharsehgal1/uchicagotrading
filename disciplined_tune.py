#!/usr/bin/env python3
"""Legacy entry point. The pfo stack uses ``eval_robust.py`` for robustness metrics."""

from __future__ import annotations

import sys

if __name__ == "__main__":
    print("Use: python eval_robust.py --data-dir <folder with prices.csv meta.csv>", file=sys.stderr)
    sys.exit(2)
