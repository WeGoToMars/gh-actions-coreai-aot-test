"""Build the portable SwiftUI app and release ZIP, then launch it."""

import argparse
import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-only", action="store_true")
args = parser.parse_args()

source = Path(__file__).resolve().parent
root = source.parents[1]
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
out = root / "artifacts/swiftui"
out.mkdir(parents=True, exist_ok=True)
app = out / "Tinkerbox.app"
release = out / f"Tinkerbox-ANE-{version}-macos-arm64.zip"

uv_version = "0.11.28"
archive = out / f"uv-{uv_version}-aarch64-apple-darwin.tar.gz"
if not archive.exists():
    download = archive.with_suffix(".part")
    urllib.request.urlretrieve(
        f"https://github.com/astral-sh/uv/releases/download/{uv_version}/uv-aarch64-apple-darwin.tar.gz",
        download,
    )
    download.rename(archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != "33540eb7c883ab857eff79bd5ac2aa31fe27b595abecb4a9c003a2c998447232":
    raise RuntimeError(f"uv archive checksum mismatch; remove {archive} and rebuild")

with tempfile.TemporaryDirectory(prefix=".app-", dir=out) as temporary:
    built = Path(temporary) / app.name
    contents = built / "Contents"
    binaries = contents / "MacOS"
    resources = contents / "Resources"
    binaries.mkdir(parents=True)
    resources.mkdir()
    uv = binaries / "uv"
    with tarfile.open(archive) as bundle:
        uv.write_bytes(bundle.extractfile("uv-aarch64-apple-darwin/uv").read())
    uv.chmod(0o755)
    shutil.copyfile(source / "LICENSE-uv.txt", resources / "LICENSE-uv.txt")
    shutil.copyfile(source / "worker.py", resources / "worker.py")
    icon_plist = Path(temporary) / "icon.plist"
    subprocess.run([
        "xcrun", "actool", str(source / "Tinkerbox.icon"), "--compile", str(resources),
        "--platform", "macosx", "--minimum-deployment-target", "27.0",
        "--app-icon", "Tinkerbox", "--output-partial-info-plist", str(icon_plist),
    ], check=True, stdout=subprocess.DEVNULL)
    requirements = subprocess.check_output([
        str(uv), "export", "--project", str(root), "--locked", "--no-dev",
        "--no-emit-project", "--no-hashes", "--no-header", "--no-annotate",
    ], text=True)
    (resources / "requirements.txt").write_text(f"tinkerbox-ane=={version}\n" + requirements)
    subprocess.run([
        "xcrun", "--sdk", "macosx", "swiftc", "-O", "-parse-as-library",
        "-target", "arm64-apple-macos27.0", str(source / "Demo.swift"),
        "-o", str(binaries / "TinkerboxDemo"),
    ], check=True)
    info = {
        "CFBundleIdentifier": "dev.tinkerbox.ane.demo",
        "CFBundleName": "Tinkerbox",
        "CFBundleExecutable": "TinkerboxDemo",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "27.0",
        "NSHighResolutionCapable": True,
    }
    info.update(plistlib.loads(icon_plist.read_bytes()))
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    for target in (uv, built):
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(target)], check=True)
    if app.exists():
        shutil.rmtree(app)
    built.rename(app)

subprocess.run(["/usr/bin/ditto", "-c", "-k", "--keepParent", str(app), str(release)], check=True)
print(f"App: {app}\nRelease: {release}", flush=True)
if not args.build_only:
    os.execv(str(app / "Contents/MacOS/TinkerboxDemo"), [str(app / "Contents/MacOS/TinkerboxDemo")])
