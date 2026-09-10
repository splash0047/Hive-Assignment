"""Download Customer Support on Twitter dataset (TWCS) via kagglehub or direct instructions."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def download_twcs_dataset(destination_dir: Path | None = None) -> Path:
    """Download the Customer Support on Twitter dataset from Kaggle.

    Uses kagglehub or checks existing data in data/raw/twcs.csv.
    """
    if destination_dir is None:
        destination_dir = Path(__file__).resolve().parents[1] / "data" / "raw"

    destination_dir.mkdir(parents=True, exist_ok=True)
    target_csv = destination_dir / "twcs.csv"

    if target_csv.exists():
        size_mb = target_csv.stat().st_size / (1024 * 1024)
        console.print(
            f"[bold green][OK] Dataset already present:[/] {target_csv} ({size_mb:.1f} MB)"
        )
        return target_csv

    console.print("[cyan]Downloading Customer Support on Twitter dataset via kagglehub...[/]")
    try:
        import kagglehub

        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        downloaded_dir = Path(path)
        console.print(f"[green]Downloaded to cache:[/] {downloaded_dir}")

        found_csvs = list(downloaded_dir.rglob("twcs.csv")) or list(downloaded_dir.rglob("*.csv"))
        if not found_csvs:
            raise FileNotFoundError(f"No CSV file found in downloaded path: {downloaded_dir}")

        source_csv = found_csvs[0]
        console.print(f"Copying {source_csv} to {target_csv}...")
        shutil.copyfile(source_csv, target_csv)
        console.print(f"[bold green][OK] Successfully placed dataset at:[/] {target_csv}")
        return target_csv

    except Exception as e:
        console.print(f"[bold red]Automatic download failed:[/] {e}")
        console.print(
            "\n[yellow]To download manually:[/]\n"
            "1. Visit: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
            "2. Download `twcs.csv`\n"
            f"3. Place it at: {target_csv.resolve()}\n"
        )
        return target_csv


if __name__ == "__main__":
    download_twcs_dataset()
