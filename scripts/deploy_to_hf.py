"""
Upload the API to a Hugging Face Docker Space.

Needs HF_TOKEN (write access) and HF_SPACE ("username/space-name") in the environment.
    python scripts/deploy_to_hf.py
"""
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

# source in this repo -> path in the Space
FILES = {
    "Dockerfile.api": "Dockerfile",                # Spaces always build ./Dockerfile
    "deploy/huggingface/README.md": "README.md",   # Space settings header + description
    "requirements.txt": "requirements.txt",
    "requirements-api.txt": "requirements-api.txt",
    "data/african_econ_crises.csv": "data/african_econ_crises.csv",
}
FOLDERS = ["config", "src", "models/champion"]

space = os.environ.get("HF_SPACE", "").strip()
token = os.environ.get("HF_TOKEN", "").strip()

# GitHub turns a missing secret/variable into an empty string - fail with a clear message
if "/" not in space:
    raise SystemExit(f"HF_SPACE must look like 'username/space-name', got {space!r}. "
                     "Set it in GitHub: Settings > Secrets and variables > Actions > Variables.")
if not token:
    raise SystemExit("HF_TOKEN is empty. Set it in GitHub: Settings > Secrets and variables > Actions > Secrets.")

with tempfile.TemporaryDirectory() as tmp:
    staging = Path(tmp)
    for source, target in FILES.items():
        (staging / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staging / target)
    for folder in FOLDERS:
        shutil.copytree(folder, staging / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    HfApi(token=token).upload_folder(
        folder_path=str(staging),
        repo_id=space,
        repo_type="space",
        commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}",
    )

print(f"Uploaded to https://huggingface.co/spaces/{space}")