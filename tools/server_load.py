"""Report instantaneous CPU, RAM and GPU load so long jobs can keep within a resource budget.

Usage:
    python tools/server_load.py            # one sample over 3 s
    python tools/server_load.py --gpu      # also report GPU memory in use (creates a small CUDA context)
    python tools/server_load.py --check 60 # exit 1 if total CPU or RAM use is above 60%

CPU numbers come from /proc/stat deltas (true instantaneous use), not from ps, whose %CPU is a lifetime average.
"""

import argparse
import os
import time


def _cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + vals[4]
    return sum(vals), idle


def _user_jiffies(uid):
    total = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            total += int(fields[11]) + int(fields[12])  # utime + stime
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    return total


def sample(interval=3.0):
    ncpu = os.cpu_count()
    uid = os.getuid()
    t0, i0 = _cpu_times()
    u0 = _user_jiffies(uid)
    time.sleep(interval)
    t1, i1 = _cpu_times()
    u1 = _user_jiffies(uid)
    dt = max(1, t1 - t0)
    total_pct = 100.0 * (1 - (i1 - i0) / dt)
    # jiffies are summed over all CPUs in /proc/stat; per-process jiffies are per-CPU time
    mine_pct = 100.0 * max(0, u1 - u0) / dt
    with open("/proc/meminfo") as f:
        mem = {line.split(":")[0]: int(line.split()[1]) for line in f}
    ram_pct = 100.0 * (1 - mem["MemAvailable"] / mem["MemTotal"])
    return {"ncpu": ncpu, "cpu_total_pct": total_pct, "cpu_mine_pct": mine_pct, "ram_used_pct": ram_pct,
            "load1": os.getloadavg()[0]}


def gpu_mem():
    import torch

    out = []
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        out.append((d, (total - free) / 2**30, total / 2**30))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--check", type=float, default=None, help="exit 1 if CPU or RAM percent exceeds this")
    a = ap.parse_args()
    s = sample(a.interval)
    print(f"cpu_total={s['cpu_total_pct']:.0f}% (all users)  cpu_mine={s['cpu_mine_pct']:.0f}%  "
          f"ram_used={s['ram_used_pct']:.0f}%  load1={s['load1']:.1f} on {s['ncpu']} cores")
    if a.gpu:
        for d, used, total in gpu_mem():
            print(f"gpu{d}: {used:.1f}/{total:.0f} GiB in use")
    if a.check is not None and max(s["cpu_total_pct"], s["ram_used_pct"]) > a.check:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
