"""Local scalability resources, without scientific imports."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SCALE = ROOT / "scalability"

def inside(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Resource must remain inside the repository")
    return p

def new_output(value):
    p = inside(value)
    base = inside("scalability/outputs")
    if p == base or not p.is_relative_to(base):
        raise ValueError("Use a new subdirectory of scalability/outputs")
    p.mkdir(parents=True, exist_ok=False)
    return p

def digest(path):
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()

def partition_files(split):
    m = json.loads((SCALE / "configs/data" / (split + ".json")).read_text(encoding="utf-8"))
    files = [inside(f["path"]) for f in m["files"]]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError("Missing partition data")
    return sorted(files)

def read_week(path, require_solver=False):
    path = inside(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if not d["complete"] or d.get("smoke", False):
        raise ValueError("A complete, full-week result is required")
    for name, recorded in d["artifacts"].items():
        artifact = inside(path.parent / name)
        if not artifact.is_relative_to(path.parent.resolve()):
            raise ValueError("Artifact escapes episode directory")
        if artifact.name == "solver.jsonl" and not require_solver:
            continue
        if digest(artifact) != recorded:
            raise ValueError("Artifact hash mismatch: " + str(artifact.relative_to(ROOT)) +
                             ". Solver record integrity remains pending; recorded hashes are not changed.")
    return d["metrics"]
