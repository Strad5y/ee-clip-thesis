Kontext für dieses Repo (ee-clip-thesis, Bachelorarbeit):
CLIP-artiges Two-Tower-Setup, das Earth Embeddings (TESSERA 128-dim, AlphaEarth 64-dim) 
mit Text aus OSM/Wikidata alignt. FROZEN encoders, nur MLP-Projektionsköpfe trainierbar.

SETTLED DECISIONS — bitte NICHT ändern oder "verbessern":
- Shared latent dim d=256; Qwen3-Embedding-4B als Text-Encoder, MRL-Truncation 2560->1024
- MLP-Kopf: Linear -> GELU -> Dropout -> Linear -> L2-normalize
- SigLIP-Loss (nicht InfoNCE); separate Two-Tower-Modelle pro Backbone
- z-Score nur für TESSERA (AE ist schon unit-normalisiert)
- Geometriefilter behält nur Point/Polygon/MultiPolygon; MultiPolygon NICHT explodieren
- Trainingspaar = jedes EE-Pixel, das eine POI-Geometrie schneidet; coverage in [0,1] als Gewicht

WORKFLOW: git pull am Anfang, commit+push am Ende. Vor Datei-Änderungen kurz erklären, was du vorhast.

ERSTE AUFGABE:
Führe make_tessera_tif.py mit einer MINI-BBox (-0.13, 51.50, -0.09, 51.53) aus, um die 
Kette Mosaik -> GeoTIFF zu testen. Zeig den Fortschritt. Prüfe vorher mit 
embeddings_count(...), wie viele Kacheln das sind. Bei Fehler: erklären und fixen, 
nicht abbrechen. Danach spatial_join.py gegen das erzeugte .tif laufen lassen und 
prüfen, dass emb-Dim = 128 rauskommt.