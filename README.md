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

## 2. Full d1 export and compile — running

[Run 40934866235](https://github.com/WeGoToMars/gh-actions-coreai-aot-test/actions/runs/40934866235)
checks out public Tinkerbox commit `228e7e632ca05fdda6ef71554cd672e83586a88f`,
installs its locked Python 3.12 environment, exports all nine backbone capacities
plus the vision tower, and compiles for standard M4 (`h16g`). The compiler and
watchdog are unchanged. Time, RSS, swap, disk space, compiler logs and OS events
are retained for diagnosis.

No inference correctness or performance result is implied by AOT compilation.
