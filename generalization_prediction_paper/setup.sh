#!/usr/bin/env bash
# Creates a Python 3.10 virtual environment and installs all dependencies.
# procgen requires Python <= 3.10 (no wheels for 3.11+ on PyPI).
#
# Usage:
#   bash setup.sh
#   source .venv/bin/activate
#   cd gen_aware_ppo
#   python main.py --env coinrun-vec --quick   # smoke-test

set -euo pipefail

PYTHON=""

# Try to find a Python 3.10 interpreter
for candidate in python3.10 python3.9 python3.8; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        echo "Found: $candidate  ($ver)"
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: No Python 3.8-3.10 found."
    echo "Install one via:"
    echo "  sudo apt install python3.10 python3.10-venv  (Debian/Ubuntu)"
    echo "  conda create -n genppo python=3.10           (conda)"
    echo "  pyenv install 3.10.14 && pyenv local 3.10.14 (pyenv)"
    exit 1
fi

echo ""
echo "Creating virtual environment with $PYTHON …"
"$PYTHON" -m venv .venv

echo "Activating …"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip …"
pip install --upgrade pip --quiet

echo "Installing requirements …"
pip install -r requirements.txt

echo ""
echo "Done!  Run:"
echo "  source .venv/bin/activate"
echo "  cd gen_aware_ppo"
echo "  python main.py --env coinrun-vec --quick    # quick smoke-test"
echo "See README.md for the full experiments."
