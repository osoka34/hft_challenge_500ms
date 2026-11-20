#!/usr/bin/env python3
"""Extract all local data-*.tar archives into ./data/ (paths inside tars already start with data/)."""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def archives(include_meta: bool = True) -> list[Path]:
    tars = sorted(DATA.glob("data-*.tar"))
    if not include_meta:
        tars = [p for p in tars if p.name != "data-meta.tar"]
    return tars


def extract_one(archive: Path, force: bool = False) -> dict:
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with tarfile.open(archive, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        total = len(members)
        print(f"\n=== {archive.name}: {total} files ===", flush=True)
        for i, m in enumerate(members, 1):
            out = ROOT / m.name
            if out.exists() and not force:
                n_skip += 1
            else:
                try:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(m)
                    if src is None:
                        raise RuntimeError("extractfile returned None")
                    out.write_bytes(src.read())
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    print(f"  FAIL {m.name}: {e}", flush=True)
            if i % 200 == 0 or i == total:
                elapsed = time.time() - t0
                print(
                    f"  {i}/{total} ok={n_ok} skip={n_skip} fail={n_fail} ({elapsed:.0f}s)",
                    flush=True,
                )
    return {
        "archive": archive.name,
        "ok": n_ok,
        "skip": n_skip,
        "fail": n_fail,
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.add_argument("--only", nargs="*", help="substring filter on archive name")
    p.add_argument("--no-meta", action="store_true")
    args = p.parse_args()

    tars = archives(include_meta=not args.no_meta)
    if args.only:
        tars = [t for t in tars if any(s in t.name for s in args.only)]
    if not tars:
        print("no archives matched", file=sys.stderr)
        sys.exit(1)

    print(f"ROOT={ROOT}")
    print(f"archives={len(tars)}")
    for t in tars:
        print(f"  {t.name} ({t.stat().st_size / 1e9:.2f} GB)")

    results = []
    for tar in tars:
        results.append(extract_one(tar, force=args.force))

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['archive']}: ok={r['ok']} skip={r['skip']} fail={r['fail']} in {r['sec']}s"
        )
    fails = sum(r["fail"] for r in results)
    if fails:
        sys.exit(2)
    print("done")


if __name__ == "__main__":
    main()
