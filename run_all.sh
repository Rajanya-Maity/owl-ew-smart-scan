#!/usr/bin/env bash
# One-click setup + run for the Smart Scan Strategy for Electronic Warfare project.
# SIH 2026 · PS 26055 · DRDO · Credit: R. Maity
set -e

echo "== Smart Scan Strategy for Electronic Warfare — Setup & Run =="

# 1. (Optional) create a virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "Running full pipeline (baseline + MAB + quick DQN training + periodic-lock demo)..."
python main.py --train_dqn --dqn_timesteps 30000 --demo

echo ""
echo "Pipeline complete. Results saved in ./results/"
echo "Launching the live dashboard..."
streamlit run dashboard/app.py
