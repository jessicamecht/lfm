#!/usr/bin/env bash
cat train.sh
cat environment.yaml
nvidia-smi
#conda update -n base -c defaults conda
conda env create -f environment.yaml
#conda init bash
#exec bash
source activate lfm
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install higher
nvidia-smi
nvidia-smi
python search.py --batch_size 30 --worker 1

