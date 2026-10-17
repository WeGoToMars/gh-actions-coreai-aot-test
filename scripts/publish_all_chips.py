"""Compatibility entry point; prefer python -m tinkerbox_ane.d1.publish."""

from tinkerbox_ane.d1.publish import main


if __name__ == "__main__":
    raise SystemExit(main())
