"""Run one real build stage, retaining output, time(1) usage and VM samples."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--cwd', default='source')
parser.add_argument('--timeout', type=int, default=1800)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
out = Path('diagnostics').resolve()
out.mkdir(exist_ok=True)


def tail(path, size=32768):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read(size).decode(errors='replace')


started = time.monotonic()
report = {'command': command, 'cwd': args.cwd, 'disk_before': shutil.disk_usage('.')._asdict()}
with (out / f'{args.name}.log').open('w') as log, (out / f'{args.name}-samples.jsonl').open('w') as samples:
    process = subprocess.Popen(['/usr/bin/time', '-l', *command], cwd=args.cwd,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    next_sample = 0
    output_offset = 0
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= next_sample:
                sample = {'seconds': elapsed, 'disk_free': shutil.disk_usage('.').free}
                print(f'{args.name}: {elapsed:.0f}s; free {sample["disk_free"] / 1024**3:.1f} GiB', flush=True)
                for key, cmd in [('swap', ['sysctl', 'vm.swapusage']), ('memory', ['vm_stat']),
                                 ('processes', ['ps', '-axo', 'pid,ppid,rss,pcpu,comm'])]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        sample[key] = result.stdout
                    except subprocess.TimeoutExpired:
                        sample[key] = 'diagnostic command timed out'
                print(sample['swap'].strip(), flush=True)
                rows = sample['processes'].splitlines()[1:]
                rows = sorted((r for r in rows if len(r.split()) >= 5),
                              key=lambda r: int(r.split()[2]), reverse=True)[:8]
                print('PID PPID RSS_KiB CPU COMMAND\n' + '\n'.join(rows), flush=True)
                samples.write(json.dumps(sample) + '\n')
                samples.flush()
                with (out / f'{args.name}.log').open('rb') as output:
                    output.seek(output_offset)
                    fresh = output.read(65536)
                    output_offset = output.tell()
                if fresh:
                    print(fresh.decode(errors='replace'), flush=True)
                else:
                    print('(no new model output since previous sample)', flush=True)
                next_sample = elapsed + 15
            if elapsed > args.timeout:
                report['timeout'] = args.timeout
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        report.update(returncode=process.returncode, seconds=time.monotonic() - started,
                      disk_after=shutil.disk_usage('.')._asdict())
        (out / f'{args.name}.json').write_text(json.dumps(report, indent=2) + '\n')
print(tail(out / f'{args.name}.log'), flush=True)
raise SystemExit(0 if report['returncode'] == 0 and 'timeout' not in report else 1)
