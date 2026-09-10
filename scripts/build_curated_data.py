"""Build curated brand pairs and historical retrieval corpus from real TWCS dataset."""

from __future__ import annotations

from rich.console import Console

from hiver_agent.config import load_config
from hiver_agent.data.sample import build_curated_data

console = Console()


def main() -> None:
    """Reconstruct Spotify threads from TWCS and generate curated datasets."""
    cfg = load_config()
    console.print("[bold cyan]Building curated datasets from real TWCS data...[/]")
    build_curated_data(cfg)


if __name__ == "__main__":
    main()
