
# AIVolumeNet: Phase 0 Repo Initialization

This archive contains Phase-0 (Project Setup) artifacts and scripts for the
AIVolumeNet project. Use the `setup_project.sh` script to initialize the
repository locally and prepare DVC/MLflow integration.

## Contents
- setup_project.sh       : One-shot setup script (creates venv, installs deps, init dvc)
- requirements.txt       : Python dependencies
- environment.yml        : Conda environment definition (optional)
- .gitignore             : Standard ignores for Python + ML
- .github/workflows/ci.yaml : Basic CI workflow for linting/install
- dvc.yaml               : sample dvc pipeline file (placeholder)
- mlflow_config.yaml     : sample mlflow config
- docs/Phase0_ProjectSetup.docx : Phase 0 doc
- docs/AIVolumeNet_Project_Proposal.pdf : Proposal PDF
- diagrams/AIVolumeNet_workflow_diagram.png : Workflow diagram PNG

## Quickstart (after cloning your GitHub repo)
1. Copy the files from this archive into your repository root.
2. Make the setup script executable: `chmod +x setup_project.sh`
3. Run the setup script: `./setup_project.sh`
4. Follow printed instructions for DVC remote configuration and MLflow setup.

