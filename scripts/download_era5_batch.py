"""Batch ERA5 downloader for the weather-pipeline completion (Phase 1D-C3).

Idempotent monthly loop over 2020-2025 using the existing single-month
framework in scripts/download_era5.py. Skips months whose NetCDF already
exists AND passes a readability check. Never fabricates data.

Modes (same landing zone data/raw/weather/era5/monthly/):
  full : era5_pjm_<Y>_<M>.nc with the 5 instantaneous v1-spec variables
         (2021-2025). Uses configs/era5_request.json (proven 2020
         pattern; ssrd is NEVER mixed into this request because the mixed
         instantaneous+accumulated CDS conversion is unreliable).
  ssrd : era5_pjm_<Y>_<M>_ssrd.nc with surface_solar_radiation_downwards
         only (supplement for ANY year; merged with 5-var files at
         processing time, raw files untouched).

Quarantine rule: a freshly downloaded file that fails validation is
RENAMED to <stem>.quarantine.nc (never deleted) with the reason logged.

On 403 cost-limit rejection the month is split into two half-month
(day-range) requests and concatenated locally with xarray.

Logging:
  - per-month request archive  request_<Y>_<M>[_ssrd].json (existing helper)
  - per-month metadata         metadata_<Y>_<M>[_ssrd].json
  - per-month checksum         sha256_<Y>_<M>[_ssrd].txt
  - run log (JSON lines)       monthly/batch_download_log.jsonl

Usage (from repo root):
    python scripts/download_era5_batch.py --mode ssrd --years 2020
    python scripts/download_era5_batch.py --mode full --years 2021 2022 2023 2024 2025
    python scripts/download_era5_batch.py --mode full --years 2021 --months 01 02

Exit codes: 0 = all requested months present+valid, 1 = some missing/failed
(details in the log; failed months are retried on the next invocation).
Secrets: never hard-coded; cdsapi reads ~/.cdsapirc (or CDSAPI_URL/KEY).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.download_era5 import (  # noqa: E402
    archive_request,
    build_cds_request,
    load_spec,
    sha256_file,
)

MONTHLY_DIR = REPO_ROOT / "data" / "raw" / "weather" / "era5" / "monthly"
V2_SPEC = REPO_ROOT / "configs" / "era5_request_v2.json"
V1_SPEC = REPO_ROOT / "configs" / "era5_request.json"
LOG_PATH = MONTHLY_DIR / "batch_download_log.jsonl"
SSRD_VAR = "surface_solar_radiation_downwards"
# Single retry policy: initial attempt + exactly one retry on validation failure.
MAX_ATTEMPTS = 2
BACKOFF_S = (60, 300)
LOCK_PATH = MONTHLY_DIR / ".batch_download.lock"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(record: dict) -> None:
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[batch] {record.get('month_key')} :: {record.get('status')} :: "
          f"{record.get('detail', '')}", flush=True)


def validate_nc(path: Path, expected_vars: list[str]) -> tuple[bool, str]:
    """Validate downloaded NetCDF. No side effects (no delete here).

    Checks (in order):
      1. file opens with xarray (tries default, then netcdf4, then scipy)
      2. required variables exist
      3. time dimension exists (valid_time/time, non-empty, hourly contiguous)
    Returns (ok, detail) where detail is the failure reason on error.
    """
    try:
        import xarray as xr
    except ImportError as e:
        return False, f"xarray missing: {e}"
    if not path.exists():
        return False, "missing file after retrieve"
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"stat failed: {e}"
    if size == 0:
        return False, "empty file (0 bytes) after retrieve"
    # 1. file opens with xarray (engine fallback for clear diagnostics)
    ds = None
    open_errs: list[str] = []
    for engine in (None, "netcdf4", "scipy"):
        try:
            if engine is None:
                ds = xr.open_dataset(path)
            else:
                ds = xr.open_dataset(path, engine=engine)
            break
        except Exception as e:
            open_errs.append(f"{engine or 'default'}:{type(e).__name__}")
            continue
    if ds is None:
        try:
            with path.open("rb") as f:
                magic = f.read(8)
            magic_info = f" size={size} magic={magic!r}"
        except Exception:
            magic_info = f" size={size}"
        return False, f"open failed with xarray ({'; '.join(open_errs)}){magic_info}"
    try:
        with ds:
            # 2. required variables exist
            vars_now = set(ds.data_vars)
            missing = [v for v in expected_vars if v not in vars_now]
            if missing:
                return False, f"missing variables {missing} (have {sorted(vars_now)})"
            # 3. time dimension exists
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            if tname not in ds.coords:
                return False, f"no valid_time/time coordinate (coords={sorted(map(str, ds.coords))})"
            times = ds[tname].to_index()
            if len(times) == 0:
                return False, "empty time axis"
            diffs = times.to_series().diff().dropna().unique()
            if len(diffs) != 1 or diffs[0] != __import__("pandas").Timedelta("1h"):
                return False, f"non-hourly steps: {diffs}"
            nan_total = int(sum(float(ds[v].isnull().sum()) for v in expected_vars))
            return True, f"ok vars={sorted(vars_now)} n={len(times)} nan={nan_total}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def write_metadata(meta_path: Path, dataset: str, variables: list[str],
                   year: str, month: str | None, target: Path,
                   request_id: str, mode: str, spec_label: str) -> None:
    import xarray as xr
    with xr.open_dataset(target) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        times = ds[tname].to_index()
        meta = {
            "dataset": dataset,
            "product_type": "reanalysis",
            "variables": variables,
            "year": year,
            "month": month,
            "bbox_requested": {"north": 48.5, "west": -92.0,
                               "south": 33.5, "east": -73.5},
            "time_coverage": {"start": str(times[0]), "end": str(times[-1]),
                              "n_steps": len(times)},
            "file": target.name,
            "file_size_bytes": target.stat().st_size,
            "acquisition_date": utcnow(),
            "request_archive": meta_path.with_name(
                meta_path.name.replace("metadata_", "request_")).name,
            "cds_request_id": request_id,
            "mode": mode,
            "spec": spec_label,
        }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def write_checksum(sum_path: Path, *paths: Path) -> None:
    lines = [f"{sha256_file(p)}  {p.name}" for p in paths if p.exists()]
    with sum_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def spec_nse_to_cds(short: str) -> dict:
    """Map v2-spec short var names to CDS request variable names."""
    return short  # v2 spec already uses CDS names


def month_request(spec: dict, year: str, month: str,
                  variables: list[str]) -> tuple[str, dict]:
    dataset, req = build_cds_request(spec, year=year, month=month)
    req["variable"] = [spec_nse_to_cds(v) for v in variables]
    return dataset, req


def retrieve(client, dataset: str, req: dict, target: Path) -> str:
    """Blocking retrieve; returns best-effort request id (or 'n/a')."""
    res = client.retrieve(dataset, req, str(target))
    rid = "n/a"
    try:
        rid = getattr(res, "reply", {}).get("request_id", "n/a") or "n/a"
    except Exception:
        pass
    return str(rid)


def retrieve_split_half_month(client, dataset: str, req: dict,
                              target: Path) -> str:
    """Two half-month retrieves concatenated along time (403 fallback)."""
    import xarray as xr
    days = req["day"]
    halves = [sorted(d for d in days if int(d) <= 15),
              sorted(d for d in days if int(d) > 15)]
    parts = []
    try:
        for i, half in enumerate(halves):
            if not half:
                continue
            r = dict(req)
            r["day"] = half
            part = target.with_name(target.stem + f".part{i}.nc")
            client.retrieve(dataset, r, str(part))
            parts.append(part)
        with xr.open_mfdataset(sorted(str(p) for p in parts),
                               combine="by_coords") as ds:
            ds = ds.sortby("valid_time" if "valid_time" in ds.coords else "time")
            ds.to_netcdf(target)
        return "n/a (split-half-month concat)"
    finally:
        for p in parts:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def download_month(client, spec: dict, spec_path: Path, year: str,
                   month: str | None, mode: str) -> bool:
    if mode == "ssrd":
        assert month is not None
        variables = [SSRD_VAR]
        stem = f"era5_pjm_{year}_{month}_ssrd"
        expected = ["ssrd"]
    else:
        # NEVER mix ssrd (accumulated) with instantaneous vars in one CDS
        # request: the mixed GRIB->NetCDF server conversion is unreliable
        # (2021-01..03 6-var requests returned unreadable payloads).
        variables = [v for v in spec["variables"] if v != SSRD_VAR]
        if month is not None:
            stem = f"era5_pjm_{year}_{month}"
        else:
            stem = f"era5_pjm_{year}"
        expected = {"2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m",
                    "10m_u_component_of_wind": "u10",
                    "10m_v_component_of_wind": "v10",
                    "total_cloud_cover": "tcc",
                    "surface_solar_radiation_downwards": "ssrd"}
        expected = [expected[v] for v in variables if v in expected]
    month_key = f"{year}-{month or 'YEAR'}-{mode}"
    spec_label = f"configs/{spec_path.name}"
    target = MONTHLY_DIR / f"{stem}.nc" if month else \
        (REPO_ROOT / "data" / "raw" / "weather" / "era5" / f"{stem}.nc")
    out_dir = target.parent
    archive_name = f"request_{year}_{month}.json" if month else f"request_{year}.json"
    if mode == "ssrd" and month:
        archive_name = f"request_{year}_{month}_ssrd.json"
    meta_name = archive_name.replace("request_", "metadata_").replace(".json", ".json")
    sum_name = archive_name.replace("request_", "sha256_").replace(".json", ".txt")

    ok, detail = (validate_nc(target, expected) if target.exists()
                  and target.stat().st_size > 0 else (False, "absent"))
    if ok:
        log({"ts": utcnow(), "month_key": month_key, "status": "SKIP_VALID",
             "detail": detail})
        return True

    try:
        dataset, req = month_request(spec, year, month, variables) \
            if month else (build_cds_request(spec, year=year)[0],
                           build_cds_request(spec, year=year)[1])
        if not month:
            dataset, req = build_cds_request(spec, year=year)
            req["variable"] = variables
    except Exception as e:
        log({"ts": utcnow(), "month_key": month_key, "status": "SPEC_ERROR",
             "detail": f"{type(e).__name__}: {e}"})
        return False

    try:
        archive_request(out_dir, spec, dataset, req, spec_path,
                        archive_name=archive_name, year=year, month=month)
    except Exception as e:
        log({"ts": utcnow(), "month_key": month_key, "status": "ARCHIVE_ERROR",
             "detail": f"{type(e).__name__}: {e}"})
        return False

    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            rid = retrieve(client, dataset, req, target)
            if target.exists() and target.stat().st_size > 0:
                ok, detail = validate_nc(target, expected)
                if ok:
                    try:
                        write_metadata(out_dir / meta_name, dataset, variables,
                                       year, month, target, rid, mode, spec_label)
                        write_checksum(out_dir / sum_name, target,
                                       out_dir / archive_name,
                                       out_dir / meta_name)
                    except Exception as e:
                        log({"ts": utcnow(), "month_key": month_key,
                             "status": "META_ERROR",
                             "detail": f"{type(e).__name__}: {e}"})
                    log({"ts": utcnow(), "month_key": month_key,
                         "status": "SUCCESS",
                         "detail": f"{detail} bytes={target.stat().st_size} "
                                   f"attempt={attempt} rid={rid}"})
                    return True
                # Validation failed: quarantine (NEVER delete fresh downloads),
                # log reason, retry once.
                last_err = f"validation failed: {detail}"
                try:
                    q = target.with_name(target.stem + ".quarantine.nc")
                    qi = 1
                    while q.exists():
                        q = target.with_name(f"{target.stem}.quarantine{qi}.nc")
                        qi += 1
                    target.rename(q)
                    quarantined = q.name
                except OSError as e:
                    quarantined = f"(quarantine failed: {e})"
                    last_err += f" (quarantine failed: {e})"
                log({"ts": utcnow(), "month_key": month_key,
                     "status": f"ATTEMPT_{attempt}_VALIDATION_FAILED",
                     "detail": f"{last_err} quarantined={quarantined} attempt={attempt}/{MAX_ATTEMPTS}"})
            else:
                last_err = "empty/missing file after retrieve"
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    pass
                log({"ts": utcnow(), "month_key": month_key,
                     "status": f"ATTEMPT_{attempt}_VALIDATION_FAILED",
                     "detail": f"{last_err} attempt={attempt}/{MAX_ATTEMPTS}"})
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            last_err = msg
            if "403" in msg or "cost limits" in msg.lower():
                try:
                    rid = retrieve_split_half_month(client, dataset, req, target)
                    ok, detail = validate_nc(target, expected)
                    if ok:
                        write_metadata(out_dir / meta_name, dataset, variables,
                                       year, month, target, rid, mode, spec_label)
                        write_checksum(out_dir / sum_name, target,
                                       out_dir / archive_name,
                                       out_dir / meta_name)
                        log({"ts": utcnow(), "month_key": month_key,
                             "status": "SUCCESS_SPLIT",
                             "detail": f"{detail} bytes={target.stat().st_size}"})
                        return True
                    last_err = f"split validation failed: {detail}"
                except Exception as e2:
                    last_err = f"split failed: {type(e2).__name__}: {e2}"
                break
            log({"ts": utcnow(), "month_key": month_key,
                 "status": f"ATTEMPT_{attempt}_FAILED", "detail": msg})
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S[attempt - 1])
    log({"ts": utcnow(), "month_key": month_key, "status": "FAILED",
         "detail": last_err})
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idempotent ERA5 monthly batch download.")
    p.add_argument("--mode", choices=["full", "ssrd"], required=True)
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument("--months", nargs="*", default=None)
    p.add_argument("--config", default=None)
    return p.parse_args()


def acquire_lock() -> bool:
    """Single-downloader guard. Returns True if lock acquired."""
    import os
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip().split()[0])
            # Check if old PID is still a live python downloader.
            import subprocess
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"Get-Process -Id {old_pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessName"],
                    capture_output=True, text=True, timeout=10)
                if old_pid != os.getpid() and "python" in (out.stdout or "").lower():
                    print(f"[batch] ERROR: another downloader running (PID {old_pid}). "
                          f"Keep only one process.", file=sys.stderr)
                    return False
            except Exception:
                pass
        except Exception:
            pass
        try:
            LOCK_PATH.unlink()
        except OSError:
            pass
    try:
        import os as _os
        import random as _random
        _random.seed()
        time.sleep(_random.uniform(0, 5))
        if LOCK_PATH.exists():
            try:
                other = LOCK_PATH.read_text(encoding="utf-8").strip().split()[0]
                if other != str(_os.getpid()):
                    print(f"[batch] ERROR: lost startup race (lock={other}). "
                          f"Keep only one process.", file=sys.stderr)
                    return False
            except OSError:
                pass
        LOCK_PATH.write_text(f"{_os.getpid()}\n", encoding="utf-8")
        time.sleep(2)
        try:
            mine = LOCK_PATH.read_text(encoding="utf-8").strip().split()[0]
            if mine != str(_os.getpid()):
                print(f"[batch] ERROR: lost startup race (lock={mine}). "
                      f"Keep only one process.", file=sys.stderr)
                return False
        except OSError as e:
            print(f"[batch] ERROR verifying lock: {e}", file=sys.stderr)
            return False
    except OSError as e:
        print(f"[batch] ERROR acquiring lock: {e}", file=sys.stderr)
        return False
    return True


def release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> int:
    args = parse_args()
    if not acquire_lock():
        return 1
    try:
        return _main_inner(args)
    finally:
        release_lock()


def _main_inner(args: argparse.Namespace) -> int:
    spec_path = Path(args.config) if args.config else (
        V2_SPEC if args.mode == "full" else V1_SPEC)
    if not spec_path.is_absolute():
        spec_path = (Path.cwd() / spec_path).resolve()
    try:
        spec = load_spec(spec_path)
    except Exception as e:
        print(f"[batch] ERROR loading spec: {e}", file=sys.stderr)
        return 1
    try:
        import cdsapi
        client = cdsapi.Client()
    except Exception as e:
        print(f"[batch] ERROR cdsapi client: {e}", file=sys.stderr)
        return 1
    months = args.months or [f"{m:02d}" for m in range(1, 13)]
    ok_all = True
    total = 0
    for year in args.years:
        for month in months:
            total += 1
            if not download_month(client, spec, spec_path, year, month, args.mode):
                ok_all = False
    print(f"[batch] DONE mode={args.mode} months={total} all_ok={ok_all}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
