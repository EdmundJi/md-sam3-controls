#!/usr/bin/env python3
"""Run the experiment matrix in scripts/experiments.py.

    python scripts/run_all.py --dry-run          # print the commands, run nothing
    python scripts/run_all.py                    # run everything still missing
    python scripts/run_all.py --only EV17_C2 EV17_C2-random
    python scripts/run_all.py --force            # re-run even if results exist

Needs a GPU and the two external SAM3 repositories (see README). Runs that
already have a ``metrics.json`` are skipped unless ``--force`` is given, so an
interrupted sweep can simply be restarted.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import RUNS, TABLE_RUNS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="RUN",
                    help="run only these (names as in experiments.RUNS)")
    ap.add_argument("--paper-only", action="store_true",
                    help="only the runs that appear in the paper's tables")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run completed runs")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    names = list(RUNS)
    if args.paper_only:
        names = [n for n in names if n in TABLE_RUNS]
    if args.only:
        unknown = [n for n in args.only if n not in RUNS]
        if unknown:
            sys.exit(f"unknown run(s): {', '.join(unknown)}")
        names = [n for n in names if n in args.only]

    done = skipped = failed = 0
    for i, name in enumerate(names, 1):
        script, extra, env_extra = RUNS[name]
        outdir = REPO / "results" / f"{name}_seed{args.seed}"

        complete = (outdir / "metrics.json").exists() and not args.force

        cmd = [sys.executable, script,
               "--output-dir", str(outdir),
               "--seed", str(args.seed), *extra]
        env = {**os.environ, **env_extra}

        prefix = " ".join(f"{k}={v}" for k, v in env_extra.items())
        printable = (prefix + " " if prefix else "") + " ".join(cmd)
        tag = "  [have results, would skip]" if complete else ""

        if args.dry_run:
            # show the command either way -- the point of a dry run is to see
            # what would be executed, not what happens to be cached
            print(f"[{i}/{len(names)}] {name}{tag}\n    {printable}")
            continue
        if complete:
            print(f"[{i}/{len(names)}] SKIP {name} (already has metrics.json)")
            skipped += 1
            continue
        print(f"[{i}/{len(names)}] {name}\n    {printable}", flush=True)

        outdir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, cwd=REPO, env=env)
        if result.returncode == 0:
            done += 1
        else:
            failed += 1
            print(f"    !! {name} exited {result.returncode}", file=sys.stderr)

    if not args.dry_run:
        print(f"\nran {done}, skipped {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
