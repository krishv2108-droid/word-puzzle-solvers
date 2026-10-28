"""Download the ORIGINAL Wordle answer and allowed-guess lists.

Source: Wordle word-list gists by cfreshman (community-extracted from the original
Wordle site's JavaScript, Feb 2022; word lists are generally treated as public domain).
We deliberately use the original lists rather than the current NYT ones: the NYT has
since removed/added answers and enlarged the accepted-guess list, and the original
2,315-answer set is the standard benchmark in the literature.

    answers.txt  - the 2,315 original solutions (what we benchmark against)
    allowed.txt  - answers + the 10,657 non-answer words accepted as guesses (12,972 total)
"""
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://gist.githubusercontent.com/cfreshman"
ANSWERS_URL = f"{BASE}/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt"
NON_ANSWERS_URL = f"{BASE}/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt"
OUT = Path(__file__).parent / "wordle"

EXPECTED_ANSWERS = 2315
EXPECTED_ALLOWED = 12972
# Checksums of the files this script writes. The gists are not versioned URLs, so a change upstream
# would otherwise slip in unnoticed as long as the word counts still matched.
SHA256 = {
    "answers.txt": "f40e47975cf360ff9febe7d49c8e285054d8fa3fb9644cea575922060951d6ee",
    "allowed.txt": "feacc549249b5dcb43e2799fbbcda14ebc33d55904fb15d043779104e60c1579",
}


def fetch(url: str) -> list[str]:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode().split()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    answers = sorted(set(fetch(ANSWERS_URL)))
    allowed = sorted(set(answers) | set(fetch(NON_ANSWERS_URL)))
    assert all(len(w) == 5 and w.isalpha() and w.islower() for w in allowed)
    assert len(answers) == EXPECTED_ANSWERS, len(answers)
    assert len(allowed) == EXPECTED_ALLOWED, len(allowed)
    for name, words in (("answers.txt", answers), ("allowed.txt", allowed)):
        text = "\n".join(words) + "\n"
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != SHA256[name]:
            raise SystemExit(f"checksum mismatch for {name}: the upstream word list changed "
                             f"(expected {SHA256[name][:12]}..., got {digest[:12]}...)")
        (OUT / name).write_text(text)
    print(f"answers: {len(answers)}, allowed guesses: {len(allowed)}")


if __name__ == "__main__":
    main()
