#!/usr/bin/env bash
# ==========================================
# Setup SafeSight environment (WSL / Linux)
# ==========================================

# 1️⃣ Désactiver conda si nécessaire
echo "[*] Deactivating conda..."
conda deactivate 2>/dev/null || true

# 2️⃣ Définir le chemin du venv Poetry
POETRY_VENV_PATH="$HOME/.cache/pypoetry/virtualenvs/safesight-Vu_VAudn-py3.11"

if [ ! -d "$POETRY_VENV_PATH" ]; then
    echo "[!] Poetry venv not found. Please create it with 'poetry install' first."
    exit 1
fi

# 3️⃣ Activer le venv Poetry
echo "[*] Activating Poetry venv..."
export PATH="$POETRY_VENV_PATH/bin:$PATH"
python --version

# 4️⃣ Ajouter MiDaS au PYTHONPATH
MIDAS_PATH="$(pwd)/libs/MiDaS"
if [ ! -d "$MIDAS_PATH" ]; then
    echo "[*] Cloning MiDaS..."
    git clone https://github.com/isl-org/MiDaS.git "$MIDAS_PATH"
fi
export PYTHONPATH="$PYTHONPATH:$MIDAS_PATH"
echo "[*] PYTHONPATH updated: $PYTHONPATH"

# 5️⃣ Installer SAM si nécessaire
echo "[*] Installing Segment Anything (SAM)..."
pip install --upgrade pip
pip install git+https://github.com/facebookresearch/segment-anything.git

# 6️⃣ Vérification rapide
echo "[*] Verifying installations..."
python -c "from segment_anything import sam_model_registry, SamPredictor; print('SAM OK')"
python -c "from midas.midas_net import MidasNet; from midas.transforms import Resize; print('MiDaS OK')"

echo "[*] Setup complete. You can now run your SafeSight scripts."
