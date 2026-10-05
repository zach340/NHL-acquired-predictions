"""
push_space.py
=============
Upload the app to the Hugging Face Space and remove the Space's old files.
Run `hf auth login` once first (token with Write access), then from the repo root:

    python deploy/push_space.py
"""

import os

from huggingface_hub import HfApi

SPACE = "Zach340/NHL-Player-Predictor"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only what the app needs: app.py, nhl_predictor/, the CSVs, the model caches, requirements.txt
IGNORE = ["README.md", "deploy/*", "raw_data/*", "pipeline/*", "scripts/*", "tests/*",
          ".git/*", ".github/*", ".vscode/*", ".claude/*", ".pytest_cache/*", "**/__pycache__/*",
          "*.log", "*.pdf", "dockerfile", "refresh_and_retrain.py", "shifts_cache/*"]
# Old Space files not in this upload (model_utils.py, MoneyPuck scripts, old .joblib, dockerfile).
# Files uploaded in the same commit are never deleted.
DELETE = ["*.py", "*.joblib", "dockerfile"]

api = HfApi()
print("Uploading app files ...")
api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=ROOT,
                  ignore_patterns=IGNORE, delete_patterns=DELETE,
                  commit_message="NHL API data, new models")
print("Updating Space settings (Streamlit 1.55, Python 3.12) ...")
api.upload_file(repo_id=SPACE, repo_type="space",
                path_or_fileobj=os.path.join(ROOT, "deploy", "space_README.md"),
                path_in_repo="README.md", commit_message="Streamlit 1.55, Python 3.12")
print(f"Done: https://huggingface.co/spaces/{SPACE} (rebuild takes ~5-10 min, watch the Logs tab)")
