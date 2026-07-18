#!/usr/bin/env python3
"""
validate.py - reusable validation harness for the opinion-dynamics project.

Catches the classes of bugs this project has actually hit:
  Python:
    1. py_compile: every listed file must byte-compile.
    2. NUL bytes: flag any file containing a 0x00 byte (Windows-mount padding).
    3. argparse dest consistency: every `args.NAME` access must map to a real
       argparse dest (the class of the `args.output_file` AttributeError).
    4. Duplicate top-level `def NAME` (silent shadowing / dead code).
  HTML:
    5. File ends with </html> (not truncated).
    6. No remote network dependency (must be offline-portable).
    7. Each inline <script> block passes `node --check`.
    8. No inlined content closes the <script> tag early (</script> balance).
    9. Required DOM ids referenced via getElementById() exist in the markup.

Run:  python3 tests/validate.py
Exits non-zero if any check fails.
"""
import ast
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY_FILES = [
    "scripts/opinion_dynamics_test_network_qwen.py",
    "scripts/ui_lancher11.py",
    "scripts/opinion_dynamics_v3_check.py",
    "scripts/network_models.py",
    "scripts/persona_profiles_support.py",
    "scripts/topic_persona_fields.py",
    "scripts/topic_persona_profiles.py",
    "scripts/create_prompts.py",
    "scripts/main_plot_network.py",
]
HTML_FILES = [
    "live_view_3d_prototype.html",
    "live_view_prototype.html",
]

# Every check appends (name, passed_bool, detail_lines) here.
RESULTS = []


def record(name, passed, details=None):
    RESULTS.append((name, passed, details or []))


def full(rel):
    return os.path.join(ROOT, rel)


# ---------------------------------------------------------------------------
# Python check 1: py_compile
# ---------------------------------------------------------------------------
def check_py_compile():
    import py_compile
    fails = []
    for rel in PY_FILES:
        try:
            py_compile.compile(full(rel), doraise=True)
        except py_compile.PyCompileError as e:
            fails.append(f"{rel}: {str(e).strip().splitlines()[-1]}")
        except Exception as e:  # pragma: no cover
            fails.append(f"{rel}: {type(e).__name__}: {e}")
    record("1. py_compile (all python files compile)", not fails, fails)


# ---------------------------------------------------------------------------
# Check 2: NUL bytes (python + html)
# ---------------------------------------------------------------------------
def check_nul_bytes():
    fails = []
    for rel in PY_FILES + HTML_FILES:
        with open(full(rel), "rb") as fh:
            data = fh.read()
        idx = data.find(b"\x00")
        if idx != -1:
            line = data[:idx].count(b"\n") + 1
            fails.append(f"{rel}: NUL byte at offset {idx} (~line {line})")
    record("2. No NUL bytes in any file", not fails, fails)


# ---------------------------------------------------------------------------
# Check 3: argparse dest consistency
# ---------------------------------------------------------------------------
def _dest_from_add_argument(call):
    """Replicate argparse's dest derivation for one add_argument() Call."""
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    option_strings = [a.value for a in call.args
                      if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not option_strings:
        return None
    if not option_strings[0].startswith("-"):
        return option_strings[0].replace("-", "_")
    longs = [o for o in option_strings if o.startswith("--")]
    if longs:
        return longs[0].lstrip("-").replace("-", "_")
    return option_strings[0].lstrip("-").replace("-", "_")


def check_argparse_dest():
    fails = []
    for rel in PY_FILES:
        src = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        src = src.replace("\x00", "")  # NUL padding is flagged by check 2; strip so AST works
        try:
            tree = ast.parse(src, filename=rel)
        except (SyntaxError, ValueError) as e:
            fails.append(f"{rel}: could not parse for argparse ({e})")
            continue

        dests = set()
        ns_names = set()
        has_add_argument = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr == "add_argument":
                    has_add_argument = True
                    d = _dest_from_add_argument(node)
                    if d:
                        dests.add(d)
                elif attr == "set_defaults":
                    for kw in node.keywords:
                        if kw.arg:
                            dests.add(kw.arg)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Attribute) \
                    and node.value.func.attr in ("parse_args", "parse_known_args"):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        ns_names.add(tgt.id)
                    elif isinstance(tgt, ast.Tuple):
                        if tgt.elts and isinstance(tgt.elts[0], ast.Name):
                            ns_names.add(tgt.elts[0].id)

        if not has_add_argument:
            continue
        ns_names.add("args")

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) \
                    and isinstance(node.value, ast.Name) and node.value.id in ns_names:
                name = node.attr
                if name not in dests:
                    fails.append(f"{rel}:{node.lineno}  args.{name} has no matching "
                                 f"argparse dest")
    seen, uniq = set(), []
    for f in fails:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    record("3. argparse dest consistency (args.NAME resolves)", not uniq, uniq)


# ---------------------------------------------------------------------------
# Check 4: duplicate top-level def NAME
# ---------------------------------------------------------------------------
def check_duplicate_defs():
    fails = []
    pat = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
    for rel in PY_FILES:
        src = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        seen = {}
        for m in pat.finditer(src):
            name = m.group(1)
            line = src[:m.start()].count("\n") + 1
            seen.setdefault(name, []).append(line)
        for name, lines in seen.items():
            if len(lines) > 1:
                fails.append(f"{rel}: top-level def {name}() defined "
                             f"{len(lines)}x at lines {lines}")
    record("4. No duplicate top-level def NAME", not fails, fails)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def check_html_complete():
    fails = []
    for rel in HTML_FILES:
        data = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        if not data.rstrip().endswith("</html>"):
            fails.append(f"{rel}: does not end with </html> (truncated?)")
    record("5. HTML files complete (end with </html>)", not fails, fails)


def check_html_remote_deps():
    fails = []
    remote = re.compile(
        r'(?:<script\b[^>]*\bsrc\s*=\s*["\']https?://'
        r'|<link\b[^>]*\bhref\s*=\s*["\']https?://'
        r'|url\(\s*["\']?https?://)', re.IGNORECASE)
    for rel in HTML_FILES:
        data = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        for m in remote.finditer(data):
            line = data[:m.start()].count("\n") + 1
            snippet = data[m.start():m.start() + 70].replace("\n", " ")
            fails.append(f"{rel}:{line}  remote dependency -> {snippet!r}")
    record("6. No remote network dependency (offline-portable)", not fails, fails)


def check_html_inline_js():
    fails = []
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except Exception:
        record("7. Inline <script> blocks are valid JS (node --check)", False,
               ["node not available on PATH"])
        return
    for rel in HTML_FILES:
        data = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        idx = 0
        for m in SCRIPT_RE.finditer(data):
            attrs, body = m.group(1), m.group(2)
            idx += 1
            if re.search(r'\bsrc\s*=', attrs, re.IGNORECASE):
                continue
            if not body.strip():
                continue
            line = data[:m.start()].count("\n") + 1
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                             encoding="utf-8") as tf:
                tf.write(body)
                tmp = tf.name
            try:
                r = subprocess.run(["node", "--check", tmp],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    err = (r.stderr or r.stdout).strip().splitlines()
                    msg = err[0] if err else "syntax error"
                    fails.append(f"{rel}: inline <script> #{idx} (near line {line}) "
                                 f"invalid JS -> {msg}")
            finally:
                os.unlink(tmp)
    record("7. Inline <script> blocks are valid JS (node --check)", not fails, fails)


def check_html_early_close():
    fails = []
    for rel in HTML_FILES:
        data = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        opens = len(re.findall(r"<script\b", data, re.IGNORECASE))
        closes = len(re.findall(r"</script\s*>", data, re.IGNORECASE))
        if opens != closes:
            fails.append(f"{rel}: <script> open/close mismatch "
                         f"(open={opens}, close={closes}) - inlined content may "
                         f"contain a literal </script>")
    record("8. No <script> closed early by inlined content", not fails, fails)


ID_ATTR_RE = re.compile(r'''\bid\s*=\s*["']([^"']+)["']''')
GETBYID_RE = re.compile(r'''getElementById\(\s*["']([^"']+)["']''')


def check_html_dom_ids():
    fails = []
    for rel in HTML_FILES:
        data = open(full(rel), "r", encoding="utf-8", errors="replace").read()
        present = set(ID_ATTR_RE.findall(data))
        referenced = sorted(set(GETBYID_RE.findall(data)))
        missing = [r for r in referenced if r not in present]
        for mid in missing:
            fails.append(f"{rel}: getElementById('{mid}') but no element with "
                         f"id=\"{mid}\" in markup")
    record("9. Required DOM ids exist (getElementById targets)", not fails, fails)


# ---------------------------------------------------------------------------
def main():
    check_py_compile()
    check_nul_bytes()
    check_argparse_dest()
    check_duplicate_defs()
    check_html_complete()
    check_html_remote_deps()
    check_html_inline_js()
    check_html_early_close()
    check_html_dom_ids()

    print("=" * 72)
    print(" OPINION-DYNAMICS VALIDATION HARNESS")
    print("=" * 72)
    any_fail = False
    for name, passed, details in RESULTS:
        status = "PASS" if passed else "FAIL"
        if not passed:
            any_fail = True
        print(f"[{status}] {name}")
        for d in details:
            print(f"         - {d}")
    print("-" * 72)
    total = len(RESULTS)
    npass = sum(1 for _, p, _ in RESULTS if p)
    print(f"SUMMARY: {npass}/{total} checks passed  ->  "
          f"{'ALL PASS' if not any_fail else 'FAILURES PRESENT'}")
    print("=" * 72)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
