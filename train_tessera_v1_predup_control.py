"""
train_tessera_v1_predup_control.py

Kontrolllauf (2026-09-19, Jakob): trainiert tessera_v1 ein zweites Mal auf der HEUTIGEN,
UNdeduplizierten pixels_tessera.parquet - identische Config/Seed wie
train_tessera_v1_dedup.py (configs/tessera_v1_predup_control.yaml unterscheidet sich von
configs/tessera_v1_dedup.yaml NUR im pixels_path/norm_stats_path). Damit unterscheiden sich
Kontrolllauf und Dedup-Lauf ausschliesslich in der Deduplizierung - der fuer die Arbeit
geltende Vergleich ist Kontrolllauf vs. Dedup-Lauf, NICHT alter Checkpoint vs. Dedup-Lauf
(dessen Datenherkunft ungeklaert ist, siehe MANIFEST.md).

Derselbe Seed wie train_tessera_v1_dedup.py (torch.manual_seed(0), da train.py::train() im
echten Trainingspfad keinen eigenen Seed setzt).
"""
import yaml
import torch

from train import train

CONFIG_PATH = "configs/tessera_v1_predup_control.yaml"
SEED = 0


def main():
    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED}) gesetzt (identisch zu train_tessera_v1_dedup.py)")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
