
#!/usr/bin/env bash
set -e
echo "=== AIVolumeNet Phase-0 Setup Script ==="
# create venv (python3)
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Created virtualenv at .venv"
fi
source .venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
# Initialize git if needed
if [ ! -d ".git" ]; then
  git init
  git checkout -b dev || true
  echo "Initialized empty git repo and created 'dev' branch."
fi
# Initialize DVC
if [ ! -d ".dvc" ]; then
  dvc init --no-scm || dvc init
  echo "Initialized DVC."
fi
echo ""
echo "PHASE-0 setup complete. Next steps (manual):"
echo "1) Configure DVC remote: dvc remote add -d storage s3://your-bucket/path"
echo "2) Start MLflow server if you want remote tracking: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &"
echo "3) Review docs/Phase0_ProjectSetup.docx and docs/AIVolumeNet_Project_Proposal.pdf"
echo "4) Add files to your GitHub repo and push."
