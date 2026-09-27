"""Locked Q1 experiment CLI (Phase 3).

Fail-closed entrypoint. Refuses to do anything unless explicitly asked:

* ``--preflight`` — SAFE verification of the orchestration graph on real
  metadata. Trains NOTHING, registers NOTHING, modifies NO canonical
  artifact, produces NO research results.
* ``--execute`` — full locked Q1 execution (11 models x H1/H6/H24 x
  seeds). Requires ``--output-dir``. NEVER run without an independent
  runner audit first.

No ranking is produced by any mode.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("q1_experiment")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Locked Q1 experiment runner (Phase 3).")
    p.add_argument("--preflight", action="store_true", help="SAFE preflight only (no training).")
    p.add_argument("--execute", action="store_true", help="Run the locked Q1 experiment matrix.")
    p.add_argument("--output", type=str, default=None, help="Preflight report path.")
    p.add_argument("--output-dir", type=str, default=None, help="Execute run directory (required with --execute).")
    p.add_argument("--models", nargs="*", default=None, help="Explicit model subset (debugging only).")
    p.add_argument("--horizons", nargs="*", type=int, default=None, help="Explicit horizon subset (debugging only).")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed subset (debugging only).")
    p.add_argument("--allow-partial", action="store_true", help="Allow a partial manifest (debugging only, labeled).")
    args = p.parse_args(argv)

    from src.q1.runner import build_manifest, prepare_q1_data, preflight, run_manifest

    if args.preflight and args.execute:
        print("Refusing: --preflight and --execute are mutually exclusive.", file=sys.stderr)
        return 2
    if args.preflight:
        report = preflight(output_path=args.output)
        print(f"PREFLIGHT {report['preflight']}: {sum(v == 'PASS' for v in report['checks'].values())}/{len(report['checks'])} checks.")
        print("NON-RESULT. Nothing trained. Models NOT ranked.")
        return 0 if report["preflight"] == "PASS" else 1
    if args.execute:
        if not args.output_dir:
            print("Refusing: --execute requires --output-dir.", file=sys.stderr)
            return 2
        out = Path(args.output_dir)
        if out.exists() and any(out.iterdir()):
            print(f"Refusing: output dir {out} exists and is non-empty.", file=sys.stderr)
            return 2
        data = prepare_q1_data()
        specs = build_manifest(models=args.models, horizons=args.horizons, seeds=args.seeds)
        records = run_manifest(specs, data, out, allow_partial=args.allow_partial)
        print(f"Executed {len(records)} locked experiments. Models NOT ranked.")
        return 0
    p.print_help()
    print("Refusing: specify --preflight (safe) or --execute (real run).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
