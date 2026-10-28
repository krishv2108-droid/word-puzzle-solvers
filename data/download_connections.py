"""Download the NYT Connections dev dataset from Hugging Face.

Source: https://huggingface.co/datasets/eric27n/NYT-Connections (Apache-2.0; compiled from
the NYT archive, Jun 2023 - Feb 2025). The underlying puzzle content belongs to The New
York Times, so the file is saved to data/connections/ (gitignored) and must not be
committed or redistributed. Used for non-commercial research/education only.
"""
import hashlib
import urllib.request
from pathlib import Path

# Pinned to an exact dataset commit and checksum. The evaluation split (which 150 puzzles are the
# held-out test set) is derived from this file, so it must not change underneath the results.
REVISION = "01d8d21680ff25207329369836bf198b4571dfd4"
SHA256 = "11500b61905783c7cc0377d94193901565de21e50027a5b7116dc2635a2926e7"
URL = f"https://huggingface.co/datasets/eric27n/NYT-Connections/resolve/{REVISION}/Connections_Data.csv"
OUT = Path(__file__).parent / "connections" / "Connections_Data.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch for {URL}\n  expected {SHA256}\n  got      {digest}\n"
                         "The upstream file changed; refusing to overwrite the evaluation data.")
    OUT.write_bytes(data)
    print(f"saved {OUT} ({OUT.stat().st_size / 1000:.0f} kB, sha256 verified)")


if __name__ == "__main__":
    main()
