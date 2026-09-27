"""ERA5 raw weather data acquisition (Phase 1D-C2-C).

Reproducible CDS API download for the PJM footprint.

- Loads the request specification from configs/era5_request.json
- Builds the CDS request dict (no secrets hard-coded; cdsapi reads ~/.cdsapirc)
- Retrieves dataset `reanalysis-era5-single-levels` to
  data/raw/weather/era5/era5_pjm_2020_2025.nc (full period),
  data/raw/weather/era5/era5_pjm_<year>.nc when --year is given
  (yearly-split strategy, docs/ERA5_YEARLY_ACQUISITION_PLAN.md), or
  data/raw/weather/era5/monthly/era5_pjm_<year>_<month>.nc when
  --year and --month are given
  (monthly-split strategy, docs/ERA5_MONTHLY_ACQUISITION_PLAN.md).
- Archives the executed request to data/raw/weather/era5/request.json
  (full period), data/raw/weather/era5/request_<year>.json (yearly),
  or data/raw/weather/era5/monthly/request_<year>_<month>.json (monthly).

Usage (from repo root):
    python scripts/download_era5.py
    python scripts/download_era5.py --config configs/era5_request.json
    python scripts/download_era5.py --year 2020
    python scripts/download_era5.py --year 2020 --month 01

Exit codes:
    0 = success (real non-empty NetCDF on disk)
    1 = failure (exact error printed, no fabricated files)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "era5_request.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "weather" / "era5"
MONTHLY_DIR = DEFAULT_OUT_DIR / "monthly"
DEFAULT_FILENAME = "era5_pjm_2020_2025.nc"
REQUEST_ARCHIVE = "request.json"

EXPECTED_DATASET = "reanalysis-era5-single-levels"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Request spec not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    return spec


def build_cds_request(spec: dict, year: str | None = None,
                      month: str | None = None) -> tuple[str, dict]:
    """Translate configs/era5_request.json into (dataset, cds_request).

    If `year` is given, only that year is requested; if `month` is also
    given, only that month of that year is requested. Variables, bbox,
    days/hours stay exactly as specified.
    """
    dataset = spec.get("dataset", EXPECTED_DATASET)
    if dataset != EXPECTED_DATASET:
        raise ValueError(
            f"Unexpected dataset {dataset!r}, expected {EXPECTED_DATASET!r}"
        )

    variables = spec.get("variables")
    if not variables or len(variables) < 1:
        raise ValueError("Spec 'variables' is missing or empty.")

    area = spec.get("area", {})
    try:
        north = float(area["north"])
        west = float(area["west"])
        south = float(area["south"])
        east = float(area["east"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Spec 'area' incomplete (need north/west/south/east): {e}")

    years = spec.get("years")
    months = spec.get("months")
    days = spec.get("days")
    hours = spec.get("hours")
    for key, val in (("years", years), ("months", months),
                     ("days", days), ("hours", hours)):
        if not val:
            raise ValueError(f"Spec '{key}' is missing or empty.")

    if year is not None:
        if year not in years:
            raise ValueError(
                f"Requested year {year!r} not in spec years {years}."
            )
        years = [year]

    if month is not None:
        if year is None:
            raise ValueError("--month requires --year.")
        if month not in months:
            raise ValueError(
                f"Requested month {month!r} not in spec months {months}."
            )
        months = [month]

    fmt = spec.get("format", "netcdf")

    cds_request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [north, west, south, east],  # CDS order: N, W, S, E
        "format": fmt,
    }
    return dataset, cds_request


def archive_request(out_dir: Path, spec: dict, dataset: str,
                    cds_request: dict, config_path: Path,
                    archive_name: str = REQUEST_ARCHIVE,
                    year: str | None = None,
                    month: str | None = None) -> Path:
    """Write executed-request archive. Returns archive path."""
    config_hash = sha256_file(config_path)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "_note": "Executed request record (Phase 1D-C2-C). "
                 "CDS API called with the cds_request below.",
        "dataset": dataset,
        "product": spec.get("product"),
        "variables": spec.get("variables"),
        "variable_notes": spec.get("variable_notes", {}),
        "years": cds_request["year"],
        "months": cds_request["month"],
        "days": spec.get("days"),
        "hours": spec.get("hours"),
        "time_note": spec.get("time_note"),
        "area": spec.get("area"),
        "format": spec.get("format", "netcdf"),
        "download_target": "data/raw/weather/era5/",
        "cds_request": cds_request,
        "request_hash": config_hash,
        "created_date": created,
    }
    if year is not None:
        record["year"] = year
    if month is not None:
        record["month"] = month
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / archive_name
    with archive_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ERA5 PJM 2020-2025 via CDS API.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to era5_request.json spec")
    parser.add_argument("--output", type=str, default=None,
                        help="Target NetCDF path (default: "
                             "monthly/era5_pjm_<year>_<month>.nc with "
                             "--year --month, era5_pjm_<year>.nc with "
                             "--year, else era5_pjm_2020_2025.nc)")
    parser.add_argument("--year", type=str, default=None,
                        help="Single year to download (e.g. 2020). "
                             "Must be listed in the spec 'years'.")
    parser.add_argument("--month", type=str, default=None,
                        help="Single month to download (e.g. 01). "
                             "Requires --year. Must be 01-12 and listed "
                             "in the spec 'months'.")
    args = parser.parse_args()

    year = args.year
    month = args.month
    if year is not None and not (len(year) == 4 and year.isdigit()):
        print(f"[era5] ERROR: --year must be a 4-digit year, got {year!r}.",
              file=sys.stderr)
        return 1
    if month is not None:
        if year is None:
            print("[era5] ERROR: --month requires --year.",
                  file=sys.stderr)
            return 1
        if not (len(month) == 2 and month.isdigit()
                and 1 <= int(month) <= 12):
            print(f"[era5] ERROR: --month must be 01-12, got {month!r}.",
                  file=sys.stderr)
            return 1

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if args.output is not None:
        target = Path(args.output)
        out_dir = target.parent
        if month is not None:
            archive_name = f"request_{year}_{month}.json"
        elif year is not None:
            archive_name = f"request_{year}.json"
        else:
            archive_name = REQUEST_ARCHIVE
    elif month is not None:
        out_dir = MONTHLY_DIR
        target = out_dir / f"era5_pjm_{year}_{month}.nc"
        archive_name = f"request_{year}_{month}.json"
    elif year is not None:
        out_dir = DEFAULT_OUT_DIR
        target = out_dir / f"era5_pjm_{year}.nc"
        archive_name = f"request_{year}.json"
    else:
        out_dir = DEFAULT_OUT_DIR
        target = out_dir / DEFAULT_FILENAME
        archive_name = REQUEST_ARCHIVE
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
        out_dir = target.parent

    print(f"[era5] Spec: {config_path}")
    print(f"[era5] Target: {target}")

    try:
        spec = load_spec(config_path)
    except Exception as e:
        print(f"[era5] ERROR loading spec: {e}", file=sys.stderr)
        return 1

    try:
        dataset, cds_request = build_cds_request(spec, year=year, month=month)
    except Exception as e:
        print(f"[era5] ERROR building CDS request: {e}", file=sys.stderr)
        return 1

    print(f"[era5] Dataset: {dataset}")
    print(f"[era5] Year: {year if year is not None else 'full period'}")
    print(f"[era5] Month: {month if month is not None else 'all'}")
    print(f"[era5] Variables: {cds_request['variable']}")
    print(f"[era5] Area (N,W,S,E): {cds_request['area']}")
    print(f"[era5] Years: {cds_request['year'][0]}..{cds_request['year'][-1]} "
          f"({len(cds_request['year'])} yrs), hourly x{len(cds_request['time'])} steps/day")

    # Secrets are never hard-coded: cdsapi reads ~/.cdsapirc (or CDSAPI_URL/KEY env).
    try:
        import cdsapi
    except ImportError as e:
        print(f"[era5] ERROR: cdsapi not installed: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = cdsapi.Client()
    except Exception as e:
        print(f"[era5] ERROR initialising CDS client: {e}", file=sys.stderr)
        print("[era5] Hint: verify ~/.cdsapirc exists with 'url' and 'key' entries.", file=sys.stderr)
        return 1

    # Archive the executed request BEFORE retrieval so the record exists
    # even if the transfer fails (proves what was attempted).
    try:
        archive_path = archive_request(out_dir, spec, dataset, cds_request,
                                       config_path, archive_name=archive_name,
                                       year=year, month=month)
        print(f"[era5] Request archived: {archive_path}")
    except Exception as e:
        print(f"[era5] ERROR writing request archive: {e}", file=sys.stderr)
        return 1

    print("[era5] Submitting CDS retrieve request (this may take a long time)...")
    try:
        client.retrieve(dataset, cds_request, str(target))
    except Exception as e:
        print(f"[era5] ERROR during CDS retrieve: {type(e).__name__}: {e}", file=sys.stderr)
        # Remove zero-byte partial file so no fabricated/empty file is mistaken for data.
        try:
            if target.exists() and target.stat().st_size == 0:
                target.unlink()
                print("[era5] Removed zero-byte partial file.", file=sys.stderr)
        except OSError:
            pass
        return 1

    if not target.exists():
        print(f"[era5] ERROR: retrieve returned but file missing: {target}", file=sys.stderr)
        return 1
    size = target.stat().st_size
    if size == 0:
        print(f"[era5] ERROR: downloaded file is empty (0 bytes): {target}", file=sys.stderr)
        return 1

    print(f"[era5] SUCCESS: {target} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
