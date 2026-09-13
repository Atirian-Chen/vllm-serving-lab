"""Archive selected measurements losslessly with checksums; no KV tensors."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new archive directory")
    args.output.mkdir(parents=True)
    manifest = []
    hashes = {}
    for stage in ("stage1", "stage2", "stage3"):
        root = getattr(args, stage)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".log", ".gz"}:
                continue
            raw = path.read_bytes()
            destination = args.output / stage / path.relative_to(root)
            if path.suffix != ".gz":
                destination = destination.with_suffix(destination.suffix + ".gz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw if path.suffix == ".gz" else gzip.compress(raw, mtime=0))
            manifest.append(dict(path=destination.relative_to(args.output).as_posix(),
                                 source_sha256=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw)))
            if path.name.startswith("run") and path.suffix == ".json":
                payload = json.loads(raw)
                if "study" in payload and payload["study"]["scenario"] == "gap0_bg2":
                    study = payload["study"]
                    key = f"{study['mode']}/repeat{study['repeat']}"
                    hashes.setdefault(key, set()).add(study["prompt_sha256"])
    if not all(len(values) == 1 for values in hashes.values()):
        raise ValueError("Workload changed across stages; archive retained for inspection")
    project = Path(__file__).resolve().parents[1]
    sources = [*project.glob("src/vllm_serving_lab/*.py"), *project.glob("scripts/*.ps1"), *project.glob("scripts/*.py")]
    source_hashes = {p.relative_to(project).as_posix():hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest() for p in sources}
    (args.output / "manifest.json").write_text(json.dumps(dict(files=manifest,
        execution_note="Measurements ran in a dirty checkout; these are the final reproduction source hashes, not a claim that every runner byte stayed unchanged during reporting.",
        source_hash_encoding="UTF-8 with line endings normalized to LF",
        published_source_sha256=source_hashes,
        same_prompts_across_stages=True, workload_hashes={k:next(iter(v)) for k,v in hashes.items()}),
        indent=2), encoding="utf-8")
    print(f"Archived {len(manifest)} files; cross-stage prompt hashes match")


if __name__ == "__main__":
    main()
