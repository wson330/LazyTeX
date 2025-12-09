# LazyTEX
## Project structure
```bash
├── config/         # config for TAMER hyperparameter
├── data/
│   └── crohme      # CROHME Dataset ; you need to download and locate in the proper directory by yourself
├── eval/             # evaluation scripts
├── tamer               # model definition folder
├── lightning_logs      # training logs
│   └── version_5     # ckpt(with fusion) for CROHME dataset
│       ├── checkpoints
│       │   └── epoch=19-step=47199-val_ExpRate=0.1917.ckpt
│       ├── config.yaml
│       └── hparams.yaml
│
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd tamer_fusion

# ----------------------------
# 1) Create Conda Environment
# ----------------------------
conda create -y -n TAMER_fusion python=3.8
conda activate TAMER_fusion

# ----------------------------
# 2) Install PyTorch (GPU / CUDA)
#    - torch 1.8.1
#    - torchvision 0.9.1
#    - CUDA 11.1
# ----------------------------
conda install -y pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 -c pytorch -c nvidia

# ----------------------------
# 3) Python dependencies
#    (listed in requirements.txt)
# ----------------------------
pip install -r requirements.txt

# ----------------------------
# 4) Editable install (TAMER package)
# ----------------------------
pip install -e .

 ```
## Dataset Preparation
We have prepared the CROHME dataset in [download link](https://disk.pku.edu.cn/link/AAF10CCC4D539543F68847A9010C607139). After downloading, please extract it to the `data/` folder.

## Training on CROHME Dataset
Next, navigate to TAMER folder and run `train.py`.
```bash
# train TAMER model on CROHME dataset
python -u train.py --config config/crohme.yaml
```

```bash
# For CROHME Dataset
bash eval/eval_crohme.sh 0

```

## Reference
- [CoMER](https://github.com/Green-Wood/CoMER) | [arXiv](https://arxiv.org/abs/2207.04410)
- [ICAL](https://github.com/qingzhenduyu/ICAL) | [arXiv](https://arxiv.org/abs/2405.09032)
- [BTTR](https://github.com/Green-Wood/BTTR) | [arXiv](https://arxiv.org/abs/2105.02412)
- [TreeDecoder](https://github.com/JianshuZhang/TreeDecoder)
- [CAN](https://github.com/LBH1024/CAN) | [arXiv](https://arxiv.org/abs/2207.11463)
- [TAMER](https://github.com/qingzhenduyu/TAMER) | [arXiv](https://arxiv.org/abs/2408.08578)