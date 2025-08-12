#!/usr/bin/env python3
"""
Summarize vm_stats.jsonl with pandas.

- Works with the logger you built (psutil-based):
  * Handles aggregated or per-NIC `net_io_per_sec`
  * Handles per-mount `nfs_io_per_sec`
- Produces windowed summary for CPU, memory, swap, net IO and per-mount NFS IO
- Prints totals and burstiness/cadence metrics for disk, net and NFS throughput.

Requires: pandas, numpy
Python: 3.9+
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import re

# -------------------------
# CONFIG
# -------------------------
STATS_FILE = "vm_stats.jsonl"          # input snapshots (JSON Lines)
WINDOW_MINUTES = None                    # summarize last N minutes; set to None for entire file
OUT_JSON = "vm_summary_pandas.json"
OUT_CSV = "vm_summary_pandas.csv"

# Thresholds used for %time-above and longest-streak
THRESHOLDS = {
    "CPU %": 65.0,
    "Memory %": 50.0,
    "Swap %": 1.0,
    "Disk read (B/s)": 0.0,
    "Disk write (B/s)": 0.0,
    "Net send (B/s)": 0.0,
    "Net recv (B/s)": 0.0,
}

# -------------------------
# Helpers
# -------------------------
def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    # robust read; skip bad lines if file is being written
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        # fallback: manual, skipping malformed lines
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(rows)
    return df


def to_utc(ts: pd.Series) -> pd.Series:
    # Accepts strings or datetimes, returns tz-aware UTC datetimes
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    return t


def sanitize_mount(path: str) -> str:
    """Return a filesystem-safe slug for mount paths."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path.strip("/"))
    return slug or "root"


def flatten_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Flatten nested dict columns and build aggregate series for net/NFS.

    Returns:
        (df, nfs_mounts) where nfs_mounts maps slug -> original mount path.
    """
    if df.empty:
        return df, {}

    # --- Parse & index by timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    # --- Generic flattener for dict columns (index-safe)
    def flatten_dict_col(frame: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
        if col not in frame.columns:
            return frame
        # Only rows where value is a dict
        mask = frame[col].map(lambda x: isinstance(x, dict))
        if not mask.any():
            # nothing to flatten
            return frame.drop(columns=[col]) if col in frame.columns else frame
        flat = pd.json_normalize(frame.loc[mask, col])
        flat.index = frame.index[mask]          # keep original timestamps
        flat = flat.add_prefix(f"{prefix}.")    # prefix like "memory.percent"
        # join preserves alignment; rows without dict become NaN in new cols
        frame = frame.drop(columns=[col]).join(flat, how="left")
        return frame

    # Core nested objects
    for base, pref in [
        ("memory", "memory"),
        ("swap", "swap"),
        ("disk", "disk"),
        ("disk_io_per_sec", "disk_io_per_sec"),
    ]:
        df = flatten_dict_col(df, base, pref)

    # --- Network: aggregated or per-NIC dict-of-dicts
    if "net_io_per_sec" in df.columns:
        sample = df["net_io_per_sec"].dropna().iloc[0] if not df["net_io_per_sec"].dropna().empty else None
        if isinstance(sample, dict) and sample:
            # Case A: aggregated dict (values are numbers)
            if all(isinstance(v, (int, float, type(None))) for v in sample.values()):
                flat = pd.json_normalize(df["net_io_per_sec"])
                flat.index = df.index
                flat = flat.add_prefix("net.")
                df = df.drop(columns=["net_io_per_sec"]).join(flat, how="left")
            else:
                # Case B: per-NIC dict-of-dicts -> sum to aggregates
                def sum_field(d: dict, field: str) -> float:
                    if not isinstance(d, dict):
                        return np.nan
                    return float(sum((inner.get(field, 0) or 0) for inner in d.values() if isinstance(inner, dict)))
                df["net.bytes_sent"] = df["net_io_per_sec"].apply(lambda d: sum_field(d, "bytes_sent"))
                df["net.bytes_recv"] = df["net_io_per_sec"].apply(lambda d: sum_field(d, "bytes_recv"))
                df["net.errin"]      = df["net_io_per_sec"].apply(lambda d: sum_field(d, "errin"))
                df["net.errout"]     = df["net_io_per_sec"].apply(lambda d: sum_field(d, "errout"))
                df["net.dropin"]     = df["net_io_per_sec"].apply(lambda d: sum_field(d, "dropin"))
                df["net.dropout"]    = df["net_io_per_sec"].apply(lambda d: sum_field(d, "dropout"))
                # keep original per-NIC blob if you want to inspect later
                df = df.drop(columns=["net_io_per_sec"])
        else:
            # Column exists but no usable dicts -> drop to avoid confusion
            df = df.drop(columns=["net_io_per_sec"])

    # --- NFS per-mount throughput (optional field)
    nfs_mounts: Dict[str, str] = {}
    if "nfs_io_per_sec" in df.columns:
        # discover all mount points
        mounts: set[str] = set()
        for d in df["nfs_io_per_sec"].dropna():
            if isinstance(d, dict):
                mounts.update(d.keys())
        for mnt in sorted(mounts):
            slug = sanitize_mount(mnt)
            nfs_mounts[slug] = mnt
            df[f"nfs.{slug}.read_bytes"] = df["nfs_io_per_sec"].apply(
                lambda d, m=mnt: float(d.get(m, {}).get("read_bytes", np.nan)) if isinstance(d, dict) else np.nan
            )
            df[f"nfs.{slug}.write_bytes"] = df["nfs_io_per_sec"].apply(
                lambda d, m=mnt: float(d.get(m, {}).get("write_bytes", np.nan)) if isinstance(d, dict) else np.nan
            )
        df = df.drop(columns=["nfs_io_per_sec"])

    return df, nfs_mounts

def estimate_sample_dt(index: pd.DatetimeIndex) -> Optional[float]:
    if index is None or index.size < 2:
        return None
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    return float(diffs.median()) if not diffs.empty else None


def longest_streak_above(arr: np.ndarray, threshold: float) -> int:
    if arr.size == 0 or np.isnan(arr).all():
        return 0
    cond = arr > threshold
    # Identify runs of True
    # trick: where cond resets, cumulative sum increments -> group runs
    groups = np.cumsum(~cond)
    # lengths = max count per group for True values
    try:
        return int(np.max(np.bincount(groups[cond])))
    except ValueError:
        return 0


def summarize_series(name: str, s: pd.Series, threshold: Optional[float], sample_dt: Optional[float]) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna().values.astype(float)
    n = int(x.size)
    if n == 0:
        return {
            "metric": name, "count": 0, "mean": None, "median": None, "std": None, "cv": None,
            "min": None, "p90": None, "p95": None, "p99": None, "max": None,
            "time_above_pct": None, "longest_streak_samples": 0, "longest_streak_seconds": 0.0,
        }
    mean = float(np.mean(x))
    med  = float(np.median(x))
    std  = float(np.std(x, ddof=1)) if n > 1 else 0.0
    cv   = float(std / mean) if mean != 0 else None
    p90  = float(np.percentile(x, 90))
    p95  = float(np.percentile(x, 95))
    p99  = float(np.percentile(x, 99))
    mn   = float(np.min(x))
    mx   = float(np.max(x))

    if threshold is None:
        tabv = None
        streak_samples = 0
    else:
        tabv = float((x > threshold).sum() / n * 100.0)
        streak_samples = longest_streak_above(x, threshold)

    streak_seconds = float(streak_samples * sample_dt) if sample_dt else 0.0

    return {
        "metric": name, "count": n, "mean": mean, "median": med, "std": std, "cv": cv,
        "min": mn, "p90": p90, "p95": p95, "p99": p99, "max": mx,
        "time_above_pct": tabv, "longest_streak_samples": streak_samples,
        "longest_streak_seconds": streak_seconds,
    }
def window_df(df: pd.DataFrame, minutes: Optional[int]) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or minutes is None:
        return df
    return df.last(f"{minutes}min")


def build_metric_map(df: pd.DataFrame, nfs_mounts: Dict[str, str]) -> List[Tuple[str, str]]:
    """(label, column) pairs present in df."""
    base = [
        ("CPU %", "cpu_percent"),
        ("Memory %", "memory.percent"),
        ("Swap %", "swap.percent"),
        ("Net send (B/s)", "net.bytes_sent"),
        ("Net recv (B/s)", "net.bytes_recv"),
    ]
    metrics = [(lbl, col) for lbl, col in base if col in df.columns]
    for slug, mount in nfs_mounts.items():
        rb = f"nfs.{slug}.read_bytes"
        wb = f"nfs.{slug}.write_bytes"
        if rb in df.columns:
            metrics.append((f"NFS read {mount} (B/s)", rb))
        if wb in df.columns:
            metrics.append((f"NFS write {mount} (B/s)", wb))
    return metrics


# --- Human-readable formatters ---
def human_bytes(n: float) -> str:
    if n is None or pd.isna(n):
        return "—"
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:,.2f} {units[i]}"


def human_rate(n: float) -> str:
    # For bytes per second
    if n is None or pd.isna(n):
        return "—"
    n = float(n)
    units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:,.2f} {units[i]}"


def human_duration(seconds: float) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    seconds = int(seconds)
    return str(timedelta(seconds=seconds))


def print_totals(start: datetime, end: datetime, totals: List[Tuple[str, float]]) -> None:
    elapsed = (end - start).total_seconds() if start and end else None
    print("\nTotals (windowed):")
    print(f"Start:   {start.isoformat()}")
    print(f"End:     {end.isoformat()}")
    if elapsed is not None:
        print(f"Elapsed: {human_duration(elapsed)}")
    print("metric".ljust(20) + "total")
    print("-" * 40)
    for label, value in totals:
        print(label.ljust(20) + human_bytes(value))
    print("-" * 40)


def print_burstiness(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "metric", "PAR", "CV", "P95/mean", "duty_cycle_%",
        "burst_count", "avg_burst_s", "cadence_s", "cadence_strength",
    ]
    widths = [20,8,8,12,14,12,14,12,18]
    print("\nBurstiness & Cadence:")
    line = " ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-" * len(line)
    print(sep)
    print(line)
    print(sep)
    for row in rows:
        vals = [
            row.get("metric", ""),
            f"{row.get('par', float('nan')):.2f}" if row.get("par") is not None else "—",
            f"{row.get('cv', float('nan')):.2f}" if row.get("cv") is not None else "—",
            f"{row.get('p95_over_mean', float('nan')):.2f}" if row.get("p95_over_mean") is not None else "—",
            f"{row.get('duty_cycle_%', float('nan')):.2f}" if row.get("duty_cycle_%") is not None else "—",
            f"{int(row.get('burst_count', 0))}" if row.get("burst_count") is not None else "—",
            f"{row.get('avg_burst_len_s', float('nan')):.2f}" if row.get("avg_burst_len_s") is not None else "—",
            f"{row.get('cadence_s', float('nan')):.2f}" if row.get("cadence_s") is not None else "—",
            f"{row.get('cadence_strength', float('nan')):.2f}" if row.get("cadence_strength") is not None else "—",
        ]
        print(" ".join(v.ljust(w) for v, w in zip(vals, widths)))
    print(sep)

def runs_above(arr: np.ndarray, threshold: float) -> tuple[int, list[int]]:
    """Return (burst_count, list_of_burst_lengths_in_samples) for arr > threshold."""
    if arr.size == 0 or np.isnan(arr).all():
        return 0, []
    cond = arr > threshold
    bursts = []
    cur = 0
    for v in cond:
        if v:
            cur += 1
        elif cur > 0:
            bursts.append(cur)
            cur = 0
    if cur > 0:
        bursts.append(cur)
    return len(bursts), bursts


def burst_metrics(name: str, s: pd.Series, sample_dt: Optional[float]) -> dict:
    """Generic burstiness stats for a throughput-like series (bytes/sec)."""
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).values.astype(float)
    if x.size == 0:
        return {}
    mean = float(np.mean(x))
    peak = float(np.max(x))
    cv = float(np.std(x, ddof=1) / mean) if mean > 0 else None
    p95_over_mean = float(np.percentile(x, 95) / mean) if mean > 0 else None
    duty_cycle = float(100.0 * np.count_nonzero(x > 0) / x.size)

    thr = float(np.percentile(x, 90))  # dynamic burst threshold
    count, lens = runs_above(x, thr)
    avg_len_s = float(np.mean(lens) * sample_dt) if lens and sample_dt else 0.0

    return {
        f"{name}_par": (peak / mean) if mean > 0 else None,
        f"{name}_cv": cv,
        f"{name}_p95_over_mean": p95_over_mean,
        f"{name}_duty_cycle_%": duty_cycle,
        f"{name}_burst_count": int(count),
        f"{name}_avg_burst_len_s": avg_len_s,
    }


def detect_cadence(series: pd.Series, sample_dt: Optional[float]) -> dict:
    """Detect dominant periodicity using autocorrelation."""
    if sample_dt is None or not isinstance(series.index, pd.DatetimeIndex):
        return {}
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).values.astype(float)
    if len(x) < 30 or np.allclose(x, 0):
        return {}
    # Autocorrelation up to ~1 hour
    max_lag = min(len(x) // 2, int(round(3600 / sample_dt)))
    ac_values = []
    for lag in range(1, max_lag):
        xc = np.correlate(x - np.mean(x), np.roll(x - np.mean(x), -lag))[0]
        denom = np.dot(x - np.mean(x), x - np.mean(x))
        ac_values.append(xc / denom if denom != 0 else 0)
    best_lag_idx = int(np.argmax(ac_values)) + 1
    best_lag_s = best_lag_idx * sample_dt
    return {
        "cadence_detected_s": best_lag_s,
        "cadence_autocorr_strength": float(np.max(ac_values)),
    }



def print_table(df_rows: pd.DataFrame) -> None:
    cols = ["metric","count","mean","median","std","cv","min","p90","p95","p99","max","time_above_pct","longest_streak_seconds"]
    df_print = df_rows[cols].copy()

    def fmt_value(metric: str, col: str, val):
        if pd.isna(val):
            return "—"
        # Humanize for throughput metrics
        if metric.endswith("(B/s)") and col in {"mean","median","p90","p95","p99","max"}:
            return human_rate(val)
        if col == "cv":
            return f"{val:,.2f}" if val is not None else "—"
        if col in {"time_above_pct"}:
            return f"{val:,.2f}"
        if col == "longest_streak_seconds":
            return f"{val:,.1f}"
        if col in {"count"}:
            return f"{int(val)}"
        try:
            return f"{float(val):,.2f}"
        except Exception:
            return str(val)

    # widths
    widths = [20,7,12,12,12,10,12,12,12,12,12,15,20]
    headers = ["metric","count","mean","median","std","cv","min","p90","p95","p99","max","time_above_%","longest_streak (s)"]
    line = " ".join(h.ljust(w) for h,w in zip(headers,widths))
    sep  = "-" * len(line)

    print("\nSummary (windowed):")
    print(sep)
    print(line)
    print(sep)
    for _, row in df_print.iterrows():
        metric_name = row["metric"]
        vals = [fmt_value(metric_name, c, row[c]) for c in cols]
        print(" ".join(v.ljust(w) for v,w in zip(vals,widths)))
    print(sep)


def label_base(label: str) -> str:
    return (label
            .lower()
            .replace(" (b/s)", "")
            .replace(" ", "_"))

def main():
    path = Path(STATS_FILE)
    df_raw = read_jsonl(path)
    if df_raw.empty:
        print(f"No data in {path}")
        sys.exit(1)

    df, nfs_mounts = flatten_columns(df_raw)
    if df.empty:
        print("No valid rows after parsing.")
        sys.exit(1)

    # Windowing
    dfw = window_df(df, WINDOW_MINUTES)

    # Estimate sample interval (seconds)
    sample_dt = estimate_sample_dt(dfw.index)

    # Throughput metrics for extras/totals
    throughput_metrics = [
        ("Disk read (B/s)", "disk_io_per_sec.read_bytes"),
        ("Disk write (B/s)", "disk_io_per_sec.write_bytes"),
        ("Net send (B/s)", "net.bytes_sent"),
        ("Net recv (B/s)", "net.bytes_recv"),
    ]
    for slug, mount in nfs_mounts.items():
        throughput_metrics.append((f"NFS read {mount} (B/s)", f"nfs.{slug}.read_bytes"))
        throughput_metrics.append((f"NFS write {mount} (B/s)", f"nfs.{slug}.write_bytes"))

    burst_rows: List[Dict[str, Any]] = []
    totals: List[Tuple[str, float]] = []
    for label, col in throughput_metrics:
        if col in dfw.columns:
            series = dfw[col]
            b = burst_metrics(label_base(label), series, sample_dt)
            cad = detect_cadence(series, sample_dt)
            burst_rows.append({
                "metric": label,
                "par": b.get(f"{label_base(label)}_par"),
                "cv": b.get(f"{label_base(label)}_cv"),
                "p95_over_mean": b.get(f"{label_base(label)}_p95_over_mean"),
                "duty_cycle_%": b.get(f"{label_base(label)}_duty_cycle_%"),
                "burst_count": b.get(f"{label_base(label)}_burst_count"),
                "avg_burst_len_s": b.get(f"{label_base(label)}_avg_burst_len_s"),
                "cadence_s": cad.get("cadence_detected_s"),
                "cadence_strength": cad.get("cadence_autocorr_strength"),
            })
            totals.append((label, float(series.fillna(0).sum())))

    # Build metric list for summary (CPU/mem/net/nfs/swap)
    metrics = build_metric_map(dfw, nfs_mounts)

    # Summaries
    rows = []
    for label, col in metrics:
        series = dfw[col]
        th = THRESHOLDS.get(label)
        rows.append(summarize_series(label, series, th, sample_dt))

    df_rows = pd.DataFrame(rows)
    print_table(df_rows)

    # Totals window
    if isinstance(dfw.index, pd.DatetimeIndex) and not dfw.index.empty:
        start, end = dfw.index.min(), dfw.index.max()
        print_totals(start, end, totals)

    # Burstiness table
    print_burstiness(burst_rows)

    # Save JSON + CSV with extras
    output = {
        "window_minutes": WINDOW_MINUTES,
        "sample_interval_seconds_est": sample_dt,
        "metrics": rows,
        "totals": totals,
        "burstiness": burst_rows,
    }
    Path(OUT_JSON).write_text(json.dumps(output, indent=2), encoding="utf-8")
    df_rows.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_JSON} and {OUT_CSV}")

if __name__ == "__main__":
    main()
