"""Download the NYT Connections dev dataset from Hugging Face.

Source: https://huggingface.co/datasets/eric27n/NYT-Connections (Apache-2.0; compiled from
the NYT archive, Jun 2023 - Feb 2025). The underlying puzzle content belongs to The New
York Times, so the file is saved to data/connections/ (gitignored) and must not be
committed or redistributed. Used for non-commercial research/education only.
"""
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/eric27n/NYT-Connections/resolve/main/Connections_Data.csv"
OUT = Path(__file__).parent / "connections" / "Connections_Data.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        OUT.write_bytes(resp.read())
    print(f"saved {OUT} ({OUT.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
