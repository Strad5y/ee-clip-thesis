"""
train_ae_v1_dedup.py

AlphaEarth-Dedup (2026-09-21, Jakob), Schritt 2: trainiert ae_v1 mit configs/ae_v1_dedup.yaml
(identische Hyperparameter/Early-Stopping wie configs/ae_v1.yaml, nur die Datenbasis ist
pixels_ae_dedup.parquet). train.py::train() setzt im echten Trainingspfad KEINEN Seed
(siehe MANIFEST.md, Seed-Status) - dieses Wrapper-Skript setzt torch.manual_seed(0) VOR dem
Aufruf, analog train_tessera_v1_dedup.py. train.py selbst wird nicht veraendert.
"""
import yaml
import torch

from train import train

CONFIG_PATH = "configs/ae_v1_dedup.yaml"
SEED = 0


def main():
    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED}) gesetzt")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
