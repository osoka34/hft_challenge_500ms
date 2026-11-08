#!/usr/bin/env python3
"""Extract a small set of PEPE days from the local tar archives into data/raw/."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

DEFAULT_TRAIN = [f"2025-11-{d:02d}" for d in range(1, 8)]
DEFAULT_VAL = [f"2026-08-{d:02d}" for d in range(10, 16)]

ARCHIVES = {
    "1000PEPEUSDT_PERP": DATA / "data-1000PEPEUSDT_PERP.tar",
    "1000PEPEUSDC_PERP": DATA / "data-1000PEPEUSDC_PERP.tar",
}

USDT_STREAMS = ("book", "bbo", "trades", "targets")
USDC_STREAMS = ("book", "bbo", "trades")


def members_for(instrument: str, day: str) -> list[str]:
    streams = USDT_STREAMS if instrument.endswith("USDT_PERP") else USDC_STREAMS
    return [f"data/{stream}/{instrument}/{day}.parquet" for stream in streams]


def extract_days(days: list[str], force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for instrument, archive in ARCHIVES.items():
        if not archive.exists():
            raise FileNotFoundError(archive)
        wanted = []
        for day in days:
            for rel in members_for(instrument, day):
                out = RAW / rel.removeprefix("data/")
                if out.exists() and not force:
                    continue
                wanted.append(rel)
        if not wanted:
            print(f"{instrument}: already extracted for requested days")
            continue
        print(f"{instrument}: extracting {len(wanted)} files from {archive.name}")
        with tarfile.open(archive, "r:*") as tf:
            for rel in wanted:
                member = tf.getmember(rel)
                out = RAW / rel.removeprefix("data/")
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    raise RuntimeError(f"missing member {rel}")
                out.write_bytes(src.read())
                print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="*", default=DEFAULT_TRAIN)
    p.add_argument("--val", nargs="*", default=DEFAULT_VAL)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    days = list(dict.fromkeys([*args.train, *args.val]))
    extract_days(days, force=args.force)
    print(f"done: {len(days)} days → {RAW}")


if __name__ == "__main__":
    main()
