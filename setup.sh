#!/bin/bash

#python3 -m venv ~/venvs/virtues4

#source ~venvs/virtues4/bin/activate

pip install numpy
pip install pandas
pip install einops
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install biopython
pip install scikit-learn --upgrade
pip install matplotlib --upgrade
pip install seaborn
pip install xformers --index-url https://download.pytorch.org/whl/cu121 --upgrade
pip install wandb
pip install pillow
pip install umap-learn
pip install POT
pip install loguru
pip install omegaconf imblearn
