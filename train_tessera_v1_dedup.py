"""
train_tessera_v1_dedup.py

Hauptaufgabe (2026-09-18, Jakob), Schritt 2: trainiert tessera_v1 auf den deduplizierten
Paaren neu (configs/tessera_v1_dedup.yaml) - identische Config/Hyperparameter/Early-Stopping
wie der bestehende tessera_v1-Lauf, NUR die Daten aendern sich.

WICHTIGER BEFUND (vor dem Start beachtet, nicht verschwiegen): train.py::train() setzt KEINEN
expliziten Seed - torch.manual_seed() wird im Modul nur in den Smoke-Tests/im Overfit-Check
aufgerufen (Zeilen ~406/435/478), nicht im echten Trainingspfad. Der urspruengliche
tessera_v1-Lauf war also NICHT geseeded - sein exakter Zufallszustand (Gewichtsinitialisierung,
Batch-Reihenfolge) ist nicht mehr rekonstruierbar. "Identischer Seed" im Sinne einer exakten
Wiederholung ist deshalb nicht erfuellbar - dieses Skript setzt stattdessen torch.manual_seed(0)
VOR dem Aufruf von train.py::train(), dieselbe Seed-Konvention wie an anderen Stellen im Repo
(build_splits.py::SEED, build_landuse_pairs.py::SAMPLE_SEED, run_landuse_probe.py::
RANDOM_HEAD_SEED), damit DIESER Lauf reproduzierbar ist - kein Versuch, den unseeded
Original-Lauf nachzubilden. train.py selbst wird NICHT veraendert.
"""
import yaml
import torch

from train import train

CONFIG_PATH = "configs/tessera_v1_dedup.yaml"
SEED = 0


def main():
    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED}) gesetzt (Original-Lauf war unseeded, siehe Docstring)")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
