"""Compatibility entry point for the shared Core AI watchdog."""

from tinkerbox_ane.guard import disk_written, main, run, stop, workers


if __name__ == "__main__":
    raise SystemExit(main())
