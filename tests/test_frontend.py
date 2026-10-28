"""Cheap guards for the plain-JS frontend (no browser needed)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
SCRIPTS = ["app.js", "connections.js"]


def test_every_element_id_the_scripts_use_exists_in_the_page():
    # a renamed or removed element makes `$("id")` return null and the page silently stops working
    ids = set(re.findall(r'\bid="([^"]+)"', (WEB / "index.html").read_text()))
    for script in SCRIPTS:
        used = set(re.findall(r'\$\("([^"]+)"\)', (WEB / script).read_text()))
        assert not used - ids, f"{script} uses ids missing from index.html: {sorted(used - ids)}"


def test_page_ids_are_unique():
    ids = re.findall(r'\bid="([^"]+)"', (WEB / "index.html").read_text())
    assert len(ids) == len(set(ids))


def test_assistant_has_the_suggested_next_guess_controls():
    html = (WEB / "index.html").read_text()
    for needle in ('id="next"', 'id="next-tiles"', 'id="next-why"', 'id="next-use"', "Suggested next guess"):
        assert needle in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_are_syntactically_valid(script):
    result = subprocess.run(["node", "--check", str(WEB / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
