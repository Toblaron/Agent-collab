"""The chat's markdown renderer lives in index.html (no build step). Run it under Node when
available: a renderer that never returns freezes the whole page, and a saved message that
triggers it freezes the room every time it's opened."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[1] / "agent_collab" / "static" / "index.html"
node = shutil.which("node")


def run_md(tmp_path, script: str) -> str:
    src = HTML.read_text()
    js = src[src.index("function esc(s)"):src.index("function renderBody")]
    (tmp_path / "md.js").write_text(js + "\nmodule.exports = { md };\n")
    (tmp_path / "run.js").write_text('const { md } = require("./md.js");\n' + script)
    out = subprocess.run([node, "run.js"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(node is None, reason="needs node")
def test_indented_first_list_item_renders(tmp_path):
    cases = ["Here are my thoughts:\n  - point one\n  - point two", "  1. first\n  2. second", "- a\n      - deep\n    - shallow"]
    out = run_md(tmp_path, f"for (const c of {json.dumps(cases)}) console.log(md(c));")
    lines = out.strip().split("\n")
    assert "<li>point one</li><li>point two</li>" in lines[0]
    assert "<ol><li>first</li><li>second</li></ol>" in lines[1]


@pytest.mark.skipif(node is None, reason="needs node")
def test_renderer_always_returns_on_random_input(tmp_path):
    parts = ["- a", "  - b", "    - c", "1. x", "   2. y", "* z", "> q", "  > q", "```", "| a | b |", "|---|---|",
             "# h", "---", "", "text", "  text", "**b**", "\t- t", "1) p", "- [ ] t", "  1. n", "***", "  ```", "- "]
    script = f"""
const parts = {json.dumps(parts)};
let seed = 7; const rnd = n => {{ seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; }};
for (let n = 0; n < 20000; n++) {{
  const lines = []; for (let j = 0, L = 1 + rnd(8); j < L; j++) lines.push(parts[rnd(parts.length)]);
  md(lines.join("\\n"));
}}
console.log("done");
"""
    assert run_md(tmp_path, script).strip() == "done"
