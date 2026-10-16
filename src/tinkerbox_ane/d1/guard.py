"""Bound Core AI commands by time, system writes and free disk space."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def disk_written():
    result = subprocess.run(["ioreg", "-r", "-c", "IOBlockStorageDriver", "-w0", "-l"],
                            capture_output=True, text=True, check=True, timeout=5)
    values = re.findall(r'"Bytes \(Write\)"=(\d+)', result.stdout)
    if not values:
        raise RuntimeError("SSD write counter unavailable")
    return sum(map(int, values))


def workers():
    result = subprocess.run(["pgrep", "-fl", "ANE.*CompilerService.xpc/Contents/MacOS/"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode not in (0, 1):
        raise RuntimeError("cannot inventory ANE compiler workers")
    return {int(line.split()[0]): line for line in result.stdout.splitlines()
            if "/ANECompilerService.xpc/Contents/MacOS/" in line
            or "/ANELargeModelCompilerService.xpc/Contents/MacOS/" in line}


def stop(process, initial, privileged):
    owned = set(workers()) - set(initial)
    # Stop workers first: killing the client alone does not cancel their I/O.
    kill = (["sudo", "-n"] if privileged else []) + ["/bin/kill"]
    if owned:
        subprocess.run(kill + ["-STOP", *map(str, sorted(owned))], capture_output=True)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    if owned:
        subprocess.run(kill + ["-KILL", *map(str, sorted(owned))], capture_output=True)
    remaining = set(workers())
    return sorted(owned - remaining)


def run(command, output, timeout=1500):
    max_bytes = 100_000_000_000
    abort_on_log = ("ANECCompileOffline() failed", "CompilationFailure")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.with_suffix(".json").exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    initial = workers()
    if any("/ANELargeModelCompilerService.xpc/" in line for line in initial.values()):
        raise RuntimeError(f"compiler workers already active: {initial}")
    permission = subprocess.run(["sudo", "-n", "/usr/bin/true"], capture_output=True)
    privileged = permission.returncode == 0
    if not privileged:
        print("warning: sudo unavailable; watchdog limits remain active, but root-owned ANE compiler workers may survive cancellation.", file=sys.stderr, flush=True)
    before = disk_written()
    started = time.monotonic()
    report = {"command": command, "write_budget_bytes": max_bytes, "timeout_s": timeout,
              "samples": [], "abort": None, "stopped_compiler_pids": [], "preexisting_workers": initial,
              "privileged_worker_stop": privileged}
    with output.with_suffix(".log").open("w") as sink:
        process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        report["pid"] = process.pid
        authentication_checked = time.monotonic()
        diagnostic_offset = 0
        diagnostic_tail = ""
        try:
            while True:
                elapsed = time.monotonic() - started
                written = disk_written() - before
                free = shutil.disk_usage(output.parent).free
                report["samples"].append({"seconds": elapsed, "system_written_bytes": written, "free_bytes": free})
                if abort_on_log:
                    with output.with_suffix(".log").open("r", errors="replace") as diagnostics:
                        diagnostics.seek(diagnostic_offset)
                        new = diagnostics.read()
                        diagnostic_offset = diagnostics.tell()
                    scanned = diagnostic_tail + new
                    matched = next((token for token in abort_on_log if token in scanned), None)
                    diagnostic_tail = scanned[-256:]
                    if matched:
                        report["abort"] = "required ANE compilation failed; refusing GPU fallback"
                        report["abort_log_match"] = matched
                if written > max_bytes:
                    report["abort"] = "SSD write budget exceeded"
                elif free < 15 * 1024**3:
                    report["abort"] = "free disk below 15 GiB"
                elif elapsed > timeout:
                    report["abort"] = "deadline exceeded"
                if privileged and time.monotonic() - authentication_checked >= 30:
                    permission = subprocess.run(["sudo", "-n", "-v"], capture_output=True)
                    authentication_checked = time.monotonic()
                    if permission.returncode:
                        privileged = report["privileged_worker_stop"] = False
                        print("warning: sudo authentication expired; continuing with unprivileged worker cancellation.", file=sys.stderr, flush=True)
                if report["abort"]:
                    report["stopped_compiler_pids"] = stop(process, initial, privileged)
                    break
                if process.poll() is not None:
                    break
                time.sleep(.2)
            # A completed client can leave compiler workers writing: retain monitoring.
            settle = time.monotonic() + 15
            while set(workers()) - set(initial) and time.monotonic() < settle and report["abort"] is None:
                written = disk_written() - before
                if written > max_bytes:
                    report["abort"] = "SSD write budget exceeded after client exit"
                    report["stopped_compiler_pids"] = stop(process, initial, privileged)
                    break
                time.sleep(.2)
            if set(workers()) - set(initial):
                report["stopped_compiler_pids"] = stop(process, initial, privileged)
                if privileged and set(workers()) - set(initial):
                    report["abort"] = report["abort"] or "compiler workers did not stop after client exit"
            report.update(returncode=process.wait(), seconds=time.monotonic() - started,
                          system_written_bytes=disk_written() - before,
                          remaining_workers={p: line for p, line in workers().items() if p not in initial})
        except BaseException as error:
            # Losing the counter or receiving an interrupt must also stop I/O.
            report["abort"] = f"watchdog interrupted: {type(error).__name__}: {error}"
            report["stopped_compiler_pids"] = stop(process, initial, privileged)
            report.update(returncode=process.wait(), seconds=time.monotonic() - started,
                          remaining_workers={p: line for p, line in workers().items() if p not in initial})
            report["passed"] = False
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            raise
    if report["remaining_workers"]:
        print(f"warning: ANE compiler workers still running: {sorted(report['remaining_workers'])}", file=sys.stderr, flush=True)
    report["passed"] = report["abort"] is None and report["returncode"] == 0
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    result = run(command, args.output)
    raise SystemExit(0 if result["passed"] else 1)
