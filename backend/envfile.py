"""Load key=value pairs from a .env file into os.environ.

The backend read os.environ directly, so a key sitting in .env was never picked
up unless the shell had already exported it -- which is exactly the trap that
made the assistant report "not configured" while the key was on disk.

Deliberately not python-dotenv: this is ~30 lines of parsing for a file format
that is `KEY=VALUE` plus comments, and a serving image should not grow a
dependency for that.

Precedence: a variable already present in the real environment always wins, so
`LLM_API_KEY=... python main.py` still overrides the file.
"""
import os

_SEARCH = (".env", "../.env")   # backend/.env first, then the repo root


def parse(text):
    """Parse .env text into a dict. Tolerates comments, blanks and quoting."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        value = value.strip()
        # Strip one layer of matching quotes. Unquoted values may carry a
        # trailing ` # comment`, quoted ones must not be cut at a '#'.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif "#" in value:
            value = value.split("#", 1)[0].strip()
        out[key] = value
    return out


def load(paths=None, override=False):
    """Load the first .env found. Returns the names (never values) that were set."""
    here = os.path.dirname(os.path.abspath(__file__))
    applied = []
    for rel in (paths or _SEARCH):
        path = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(here, rel))
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = parse(f.read())
        except OSError:
            continue
        for k, v in data.items():
            if override or not os.environ.get(k):
                os.environ[k] = v
                applied.append(k)
        break                      # first file wins; do not merge several
    return applied


def _self_check():
    """Parsing must survive the shapes people actually write, and never log values."""
    text = """
# a comment
LLM_PROVIDER=grok
export LLM_API_KEY=xai-secret-value
QUOTED="has spaces and # hash"
SINGLE='single quoted'
TRAILING=value   # inline comment
EMPTY=
URL=https://api.x.ai/v1
EQUALS=a=b=c
  INDENTED=yes
not_a_line
123BAD=skipped
"""
    d = parse(text)
    assert d["LLM_PROVIDER"] == "grok", d
    assert d["LLM_API_KEY"] == "xai-secret-value", "export prefix not stripped"
    assert d["QUOTED"] == "has spaces and # hash", "quoted # was truncated"
    assert d["SINGLE"] == "single quoted"
    assert d["TRAILING"] == "value", "inline comment not stripped"
    assert d["EMPTY"] == ""
    assert d["URL"] == "https://api.x.ai/v1", "URL slashes mangled"
    assert d["EQUALS"] == "a=b=c", "split on the wrong ="
    assert d["INDENTED"] == "yes"
    assert "not_a_line" not in d and "123BAD" not in d

    # real environment must win over the file
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, ".env")
        with open(p, "w") as f:
            f.write("ENVFILE_TEST_A=from_file\nENVFILE_TEST_B=from_file\n")
        os.environ["ENVFILE_TEST_A"] = "from_shell"
        os.environ.pop("ENVFILE_TEST_B", None)
        load([p])
        assert os.environ["ENVFILE_TEST_A"] == "from_shell", "file overrode the shell"
        assert os.environ["ENVFILE_TEST_B"] == "from_file", "file value not applied"
        load([p], override=True)
        assert os.environ["ENVFILE_TEST_A"] == "from_file", "override=True ignored"
        for k in ("ENVFILE_TEST_A", "ENVFILE_TEST_B"):
            os.environ.pop(k, None)

    # a missing file is not an error
    assert load(["/nonexistent/.env"]) == []
    print("envfile.py self-check passed")


if __name__ == "__main__":
    _self_check()
