# data/splits/

- `splits_london.csv` — **aktuell.** Pixelbasierte Zell->Split-Zuordnung (`build_splits_pixels.py`,
  Vereinigung der Zellen aus `pixels_tessera.parquet`/`pixels_ae.parquet`). Alle laufenden
  Trainings-/Eval-Skripte lesen diese Datei.
- `splits_london_4key.csv` — **ARCHIV, nicht aktuell.** Backup der ALTEN, POI-basierten
  Zell->Split-Zuordnung (`build_splits.py`, gebaut auf den 279.988 POIs der alten 4-Key-
  Extraktion, vor der Pixel-Aggregation). `build_splits_pixels.py` hat `splits_london.csv` am
  3.9. an derselben Stelle mit der neuen, pixelbasierten Zuordnung überschrieben; diese Datei
  ist die davor gesicherte alte Version.
  **Darf nie als aktueller Split verwendet werden** — Kreuztabellen-Check zeigt nur ~65%
  Übereinstimmung mit `splits_london.csv` (nahe dem Erwartungswert zweier unabhängiger
  80/10/10-Zuordnungen). `landuse_pairs_ae.parquet`/`landuse_pairs_tessera.parquet` wurden
  versehentlich mit dieser alten Zuordnung gebaut, siehe `data/archive_gen1/` und
  `data/processed/MANIFEST.md`.
