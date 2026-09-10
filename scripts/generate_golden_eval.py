"""Generate the curated 200-example Golden Evaluation Set from real TWCS Spotify conversations."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and scripts dir to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from scripts.build_real_golden import build_real_golden  # noqa: E402


def main():
    """Build and freeze 200 genuine customer support examples from TWCS."""
    build_real_golden()


if __name__ == "__main__":
    main()
