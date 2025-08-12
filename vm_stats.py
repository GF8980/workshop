#!/usr/bin/env python3

import psutil
import time
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone

# ---------------------
# CONFIG
# ---------------------
INTERVAL_SECONDS = 1
MAX_DURATION_SECONDS = 600
STATS_LOG_FILE = "vm_stats.jsonl"
PEAKS_LOG_FILE = "vm_peaks.json"
LOG_PER_INTERFACE = False # If true, logs per-NIC stats instead of aggregated
NFS_MOUNTS = ["/sset/data"]

_BYTES_LINE = re.compile(r"^bytes:\s+(\d+)\s+(\d+)")  # read_bytes, write_bytes

def _detect_nfs_mounts() -> dict:
    """
    Returns {mountpoint: True} for mounts that are NFS/nfs4 according to /proc/mounts.
    """
    found = {}
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                device, mountpoint, fstype = parts[0], parts[1], parts[2]
                if fstype.lower().startswith("nfs"):
                    found[mountpoint] = True
    except Exception:
        pass
    return found

def nfs_counters(paths: list[str]) -> dict:
    """
    Parse /proc/self/mountstats and return:
      { mount_path: { "read_bytes": int, "write_bytes": int, "age_sec": float|None } }
    Only for the provided paths that are actually NFS mounts.
    """
    out = {}
    ms = Path("/proc/self/mountstats")
    if not ms.exists():
        return out

    want = set(paths)
    # Gate by real NFS mounts so we don't misreport local filesystems
    nfs_mounted = _detect_nfs_mounts()
    want = {p for p in want if p in nfs_mounted}

    cur_mount = None
    try:
        for raw in ms.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            # New section header
            # Example: device 10.0.0.5:/export mounted on /sset/data with fstype nfs4 statvers=1.1
            if line.startswith("device ") and " mounted on " in line and " fstype nfs" in line:
                try:
                    # Extract the mountpoint between 'mounted on ' and ' with fstype'
                    mnt = line.split(" mounted on ", 1)[1].split(" with fstype", 1)[0].strip()
                except Exception:
                    mnt = None
                cur_mount = mnt if mnt in want else None
                if cur_mount and cur_mount not in out:
                    out[cur_mount] = {"read_bytes": 0, "write_bytes": 0, "age_sec": None}
                continue

            if not cur_mount:
                continue

            if line.startswith("bytes:"):
                # "bytes: <read> <write> ..." — take first two ints
                m = _BYTES_LINE.match(line)
                if m:
                    out[cur_mount]["read_bytes"] = int(m.group(1))
                    out[cur_mount]["write_bytes"] = int(m.group(2))
            elif line.startswith("age:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        out[cur_mount]["age_sec"] = float(parts[1])
                    except Exception:
                        pass
    except Exception:
        return {}
    return out

def nfs_usage(path: str):
    """Return psutil.disk_usage(path) as dict or None on error."""
    try:
        du = psutil.disk_usage(path)
        return {"total": du.total, "used": du.used, "free": du.free, "percent": du.percent}
    except Exception:
        return None
    
# ---------------------
# INITIALIZATION
# ---------------------
start_time = time.time()
last_nfs = nfs_counters(NFS_MOUNTS)
last_disk_io = psutil.disk_io_counters()
last_net_io = psutil.net_io_counters(pernic=LOG_PER_INTERFACE)
cpu_count = psutil.cpu_count(logical=True)

# Initializae peak tracker
peaks = {}

def update_peaks(metric_name, value):
    peaks[metric_name] = max(peaks.get(metric_name, 0), value)

# ---------------------
# MONITOR
# ---------------------
with open(STATS_LOG_FILE, 'w') as stats_log:
    while time.time() - start_time < MAX_DURATION_SECONDS:
        timestamp = datetime.now(timezone.utc).isoformat()

        # --- CPU ---
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        update_peaks("cpu_percent", cpu_percent)

        # --- Memory ---
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        update_peaks("memory_percent", mem.percent)
        update_peaks("swap_percent", swap.percent)

        # --- Disk usage & I/O ---
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        disk_io = psutil.disk_io_counters()
        read_diff = disk_io.read_bytes - last_disk_io.read_bytes
        write_diff = disk_io.write_bytes - last_disk_io.write_bytes

        # Disk busy% (Linux)
        busy_time_ms = getattr(disk_io, "busy_time", None)
        disk_busy_pct = None
        if busy_time_ms is not None:
            # Calculate busy% over the last interval
            delta_busy = busy_time_ms - getattr(last_disk_io, "busy_time", busy_time_ms)
            disk_busy_pct = (delta_busy / (INTERVAL_SECONDS * 1000)) * 100
            update_peaks("disk_busy_percent", disk_busy_pct)

        update_peaks("disk_read_bytes_per_sec", read_diff)
        update_peaks("disk_write_bytes_per_sec", write_diff)

        last_disk_io = disk_io

        # --- Network I/O ---
        net_io = psutil.net_io_counters(pernic=LOG_PER_INTERFACE)
        if LOG_PER_INTERFACE:
            net_data= {}
            for nic, stats in net_io.items():
                last_stats = last_net_io.get(nic)
                if not last_stats:
                    continue
                sent_diff = stats.bytes_sent - last_stats.bytes_sent
                recv_diff = stats.bytes_recv - last_stats.bytes_recv
                errin_diff = stats.errin - last_stats.errin
                errout_diff = stats.errout - last_stats.errout
                dropin_diff = stats.dropin - last_stats.dropin
                dropout_diff = stats.dropout - last_stats.dropout

                net_data[nic] = {
                    "bytes_sent": sent_diff,
                    "bytes_recv": recv_diff,
                    "errin": errin_diff,
                    "errout": errout_diff,
                    "dropin": dropin_diff,
                    "dropout": dropout_diff
                }
        else:
            sent_diff = net_io.bytes_sent - last_net_io.bytes_sent
            recv_diff = net_io.bytes_recv - last_net_io.bytes_recv
            errin_diff = net_io.errin - last_net_io.errin
            errout_diff = net_io.errout - last_net_io.errout
            dropin_diff = net_io.dropin - last_net_io.dropin
            dropout_diff = net_io.dropout - last_net_io.dropout

            update_peaks("net_sent_bytes_per_sec", sent_diff)
            update_peaks("net_recv_bytes_per_sec", recv_diff)

            net_data = {
                "bytes_sent": sent_diff,
                "bytes_recv": recv_diff,
                "errin": errin_diff,
                "errout": errout_diff,
                "dropin": dropin_diff,
                "dropout": dropout_diff
            }

        last_net_io = net_io

        # --- Load average (Linux) ---
        try:
            load1,load5, load15 = os.getloadavg()
            load_per_core = load1 / cpu_count if cpu_count else None
            update_peaks("load1", load1)
        except OSError:
            load1 = load5 = load15 = load_per_core = None
        
        # --- NFS per-second throughput + capacity for configured mounts ---
        nfs_now = nfs_counters(NFS_MOUNTS)
        nfs_data = {}         # per-second deltas
        nfs_caps = {}         # capacity per mount

        for mnt in NFS_MOUNTS:
            cur = nfs_now.get(mnt)
            prev = last_nfs.get(mnt)

            # capacity/usage (works on NFS via statvfs)
            cap = nfs_usage(mnt)
            if cap:
                nfs_caps[mnt] = cap

            if not cur or not prev:
                # First pass or mount not present; record zeros so schema is stable
                nfs_data[mnt] = {"read_bytes": 0, "write_bytes": 0, "age_sec": cur.get("age_sec") if cur else None}
                continue

            # Per-interval deltas (bps-like values)
            rb = max(0, cur["read_bytes"]  - prev["read_bytes"])
            wb = max(0, cur["write_bytes"] - prev["write_bytes"])

            nfs_data[mnt] = {"read_bytes": rb, "write_bytes": wb, "age_sec": cur.get("age_sec")}
            # Update peaks
            update_peaks(f"nfs_read_bytes_per_sec[{mnt}]", rb)
            update_peaks(f"nfs_write_bytes_per_sec[{mnt}]", wb)

        # carry forward for next interval
        last_nfs = nfs_now


        # --- Snapshot ---
        snapshot = {
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "cpu_per_core_percent": cpu_per_core,
            "memory": {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "percent": swap.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "disk_io_per_sec": {
                "read_bytes": read_diff,
                "write_bytes": write_diff,
                "read_time_ms": getattr(disk_io, "read_time", None),
                "write_time_ms": getattr(disk_io, "write_time", None),
                "busy_percent": disk_busy_pct
            },
            "net_io_per_sec": net_data,
            "load": {
                "load1": load1,
                "load5": load5,
                "load15": load15,
                "load_per_core": load_per_core
            },
            "nfs_io_per_sec": nfs_data,        # { "/sset/data": {read_bytes, write_bytes, age_sec}, ... }
            "nfs_usage": nfs_caps        # { "/sset/data": {total, used, free, percent}, ... }
        }

        # --- Write Snapshot ---
        stats_log.write(json.dumps(snapshot) + "\n")
        stats_log.flush()

        time.sleep(INTERVAL_SECONDS)

# --- Save Peak Values ---
with open(PEAKS_LOG_FILE, 'w') as f:
    json.dump(peaks, f, indent=2)

print(f"\n Monitoring Complete.")
print(f"> Interval logs: {STATS_LOG_FILE}")
print(f"> Peak values: {PEAKS_LOG_FILE}")```
