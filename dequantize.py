"""
dequantize.py

AlphaEarth-Kacheln werden vom GCS-Bucket als quantisiertes int8 ausgeliefert, ohne
Scale/Offset-Metadaten im GeoTIFF (scale=1.0, offset=0.0 auf allen 64 Baendern - rasterio
skaliert also NICHT automatisch beim Read). Verifiziert per L2-Norm-Test ueber echte
Stichproben-Pixel aus zwei London-Kacheln (565 gueltige Samples, NoData ausgefiltert):
die Quantisierung ist NICHT die einfache lineare int8/127-Skalierung (die ergibt eine
mittlere Norm von ~2.55, klar nicht 1.0), sondern vorzeichenerhaltend quadratisch:

    x = int8 / 127.5
    embedding = sign(x) * |x|^2

(Vorwaertsrichtung/Quantisierung: sign(e) * |e|^0.5 * 127.5, geklemmt auf [-127, 127].)
Damit liegt die gemessene L2-Norm bei mean=1.0002, std=0.002 ueber beide Kacheln - stimmt
mit der GCS-README-Aussage ueberein, dass die Embeddings NACH Dequantisierung auf
euklidische Laenge 1 normiert sind.

NoData-Wert ist -128 (aus dem GeoTIFF-nodata-Tag) - Pixel mit -128 auf ALLEN Baendern werden
zu NaN statt (falsch) dequantisiert, damit sie ueber die bestehenden NaN-Checks
(validate_pairs.py) auffallen statt eine erfundene Embedding zu bekommen.

TESSERA ist NICHT betroffen - der Quantisierungsfaktor dort ist unbekannt und wird separat
geklaert (noch keine Dequantisierung implementiert, siehe spatial_join.py).
"""
import numpy as np

NODATA_AE = -128


def dequantize_ae(raw: np.ndarray) -> np.ndarray:
    """raw: int8-Array, letzte Achse = Baender (..., 64). NoData-Pixel (alle Baender == -128)
    werden zu NaN, alle anderen ueber die vorzeichenerhaltende quadratische Abbildung
    dequantisiert (siehe Modul-Docstring, per L2-Norm-Test gegen echte Kacheln verifiziert)."""
    raw = np.asarray(raw)
    is_nodata = (raw == NODATA_AE).all(axis=-1)

    x = raw.astype(np.float32) / 127.5
    e = np.sign(x) * np.abs(x) ** 2

    e[is_nodata] = np.nan
    return e


def _smoke_test():
    """Nur Shape-/NaN-Handling: zufaellige int8-Werte sind KEINE echten Embeddings, ihre Norm
    muss hier NICHT nahe 1 liegen - die eigentliche Formel-Verifikation lief gegen echte
    AlphaEarth-Kacheln (siehe Docstring), nicht gegen Zufallsdaten."""
    rng = np.random.default_rng(0)
    n, bands = 50, 64

    raw = rng.integers(-127, 128, size=(n, bands)).astype(np.int8)
    raw[0, :] = NODATA_AE  # NoData-Zeile einschleusen

    out = dequantize_ae(raw)
    assert out.shape == (n, bands)
    assert np.isnan(out[0]).all(), "NoData-Zeile sollte komplett NaN sein"
    assert np.isfinite(out[1:]).all(), "Nicht-NoData-Zeilen sollten keine NaN enthalten"
    assert (np.abs(out[1:]) <= 1.0 + 1e-6).all(), "Dequantisierte Werte sollten in [-1, 1] liegen"

    print(f"Smoke-Test OK: NoData (-128 auf allen Baendern) -> NaN, sonst endliche Werte in "
          f"[-1, 1], Shape {out.shape} erhalten.")


if __name__ == "__main__":
    _smoke_test()
