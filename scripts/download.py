"""Fetch the same pinned checkpoint separately with the standard downloader."""

from huggingface_hub import HfApi, snapshot_download
from tinkerbox_ane.d1 import REPO, REVISION

info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
for file in info.siblings:
    print(f'{file.rfilename}: {file.size} bytes', flush=True)
print(snapshot_download(REPO, revision=REVISION), flush=True)
