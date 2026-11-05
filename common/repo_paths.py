"""Repository-local resources. No imports of scientific modules."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def inside(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path must remain inside this repository")
    return path

def profile_files(split):
    names = json.loads((ROOT / "main/configs/data.json").read_text(encoding="utf-8"))[split]
    paths = [inside(Path("data/profiles") / name) for name in names]
    if len(set(paths)) != len(paths) or not all(p.is_file() for p in paths):
        raise ValueError("Missing or duplicate data profiles")
    return sorted(paths)

def new_output(value):
    path = inside(value)
    base = inside("main/outputs")
    if path == base or not path.is_relative_to(base):
        raise ValueError("Output must be a new subdirectory of main/outputs")
    path.mkdir(parents=True, exist_ok=False)
    return path
