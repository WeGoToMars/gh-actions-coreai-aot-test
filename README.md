# Core AI AOT on GitHub-hosted macOS

Step-by-step experiments with the free public-repository `xcode-27` runner.
Target: LiquidAI/d1-3B, using Tinkerbox's existing compiler, watchdog and full
ANE-placement verification. No paid runners are used.

## 1. Environment preflight — passed

[Run 40934535926](https://github.com/WeGoToMars/gh-actions-coreai-aot-test/actions/runs/40934535926)
completed successfully on October 29, 2026. Selected raw diagnostics are in
[results/01-preflight](results/01-preflight); the run artifact includes every command output.

| Property | Observed |
| --- | --- |
| Runner image | `macos27`, `20261026.0314.1` |
| Hardware | `VirtualMac2,1`, Apple M1 (Virtual), 3 CPUs |
| Memory | 7 GiB |
| OS | macOS 27.0.1, build 26A434 |
| Xcode | 27.1, build 27A9275; macOS SDK 27.0 |
| APFS volume size | 149.5 GiB |
| Free space after Metal installation | 35.2 GiB |
| SSD write counters required by existing watchdog | Available |
| Passwordless sudo | Available |
| `coreai-build` before explicit Metal installation | Already available |
| Core AI `AIModel.deviceArchitectureName` | `unknown` |

The documented 14 GB storage specification does not describe the actual free
space on this runner instance. The existing 15 GiB watchdog floor is satisfied
without deleting installed software. This is a measurement, not a guarantee for
future images.

`xcodebuild -downloadComponent MetalToolchain` succeeded. Both before and after,
`xcrun coreai-build compile --help` worked. The VM reports `unknown` as its Core AI
architecture; all AOT experiments therefore pass an explicit target architecture.

## 2. Initial full export — interrupted by experiment operator

[Run 40934866235](https://github.com/WeGoToMars/gh-actions-coreai-aot-test/actions/runs/40934866235)
checks out public Tinkerbox commit `228e7e632ca05fdda6ef71554cd672e83586a88f`,
was configured to install its locked Python 3.12 environment, export all nine
backbone capacities plus the vision tower, and compile for standard M4 (`h16g`). The compiler and
watchdog are unchanged. Time, RSS, swap, disk space, compiler logs and OS events
are retained for diagnosis.

No inference correctness or performance result is implied by AOT compilation.

The operator canceled this attempt after the live UI appeared stalled. Recovered
logs show the run was making progress: download finished in 63 seconds, all nine
capacities were staged by 374 seconds, and conversion to Core AI had begun.
There was no demonstrated export or compiler failure. Resource samples continued
every ~15 seconds; swap use reached about 2.2 GiB. The cancellation was premature.
No AOT compilation ran. See [recovered export evidence](results/02-interrupted-export).

## 3. Full export passed; stock 15 GiB floor stopped compilation

[Run 40936211783](https://github.com/WeGoToMars/gh-actions-coreai-aot-test/actions/runs/40936211783).


The follow-up keeps the standard downloader and model/compiler settings, but
separates checkpoint download from offline export, prints swap and the largest
processes in every heartbeat, and uploads diagnostics between stages. It also
retains compiled placement manifests when available.

The separate checkpoint download passed in 41.5 seconds (40.93 seconds reported
by `time`), with the standard unauthenticated downloader. It downloaded the pinned
6 GB checkpoint, left 28.5 GiB free, and used no swap at the last sampled point.
The subsequent export uses `HF_HUB_OFFLINE=1`. Partial diagnostics are available
as the `d1-download-40936211783` Actions artifact before export completed.

The full export passed in **786.9 seconds (13m 07s)**, including all nine backbone
capacities and vision. Source bundle size: **6,273,626,731 bytes**. Peak sampled
swap use was **10,831.75 MiB**. Only **12.3 GiB** remained at compilation start,
and the unchanged watchdog stopped immediately at its **15 GiB** disk floor.
This was a disk-reserve failure; full backbone AOT compilation was not tested.
[Retained reports and samples](results/03-full-default-floor).

## 4. Real d1 vision export and AOT — passed

[Run 40937819569](https://github.com/WeGoToMars/gh-actions-coreai-aot-test/actions/runs/40937819569)
used the same pinned source, model, free runner and target `h16g` (standard M4).
The real d1 vision tower exported in **54.7 seconds** and compiled and verified
in **33.5 seconds**. The existing watchdog passed with no abort, and no swap was
used in the samples. The compiler's manifest contains `mps.fullyPlacedOnANE` and
`mps.noGPUActivity`; inspect reports `supportedArchitectures: ["h16g"]`,
`supportedChips: ["M4"]`, minimum macOS 27.0, and the `main` function.
[Retained logs, watchdog report, inspect output and raw placement manifest](results/04-vision).

This proves offline cross-compilation works on the VM for this real model asset,
even though Core AI reports the host architecture as `unknown`. It does not
establish full-backbone compilation or on-device inference correctness/performance.

## 5. Full d1 with requested 5 GiB reserve — compiler timed out

[Run 40941582940](https://github.com/WeGoToMars/gh-actions-coreai-aot-test/actions/runs/40941582940).

The next trial changes only the disk floor from 15 to **5 GiB**, using the
[explicit two-line CI patch](patches/ci-five-gib-floor.patch). No unused Xcodes or
checkpoint files are removed in this trial. Time limits, the SSD write budget,
source identity checks, function/architecture/deployment validation and full ANE
placement verification remain in place. The local Tinkerbox checkout is unchanged.
Diagnostic uploads now include hidden build directories so raw watchdog reports
from incomplete bundles are retained as well.

The full export succeeded in 979 seconds. The 5 GiB reserve allowed compilation to start; the watchdog stopped it after its unchanged 600-second deadline, with approximately 17 GiB free. The full backbone remains unverified. See [the reports](results/05-five-gib-floor). The next trial raises the compiler deadline to 1,800 seconds while retaining the watchdog, 5 GiB floor, write budget and ANE verification.
