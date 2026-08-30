"""
app.py
======
Eine einfache Streamlit-Weboberfläche für den Fashion-Matching-Look-Finder.

Das Skript lädt ein vom Nutzer hochgeladenes Bild, berechnet dessen
CLIP-Embedding und zeigt die drei ähnlichsten Looks aus dem Katalog an.
"""

import json
import os

import numpy as np
import streamlit as st
from PIL import Image

# Wir importieren die bestehenden Hilfsfunktionen aus dem Projekt.
# So wird das CLIP-Modell und die Embedding-Berechnung nicht dupliziert.
from clip_utils import load_clip_model, get_image_embedding
from find_similar import find_similar_images


# ---------------------------------------------------------------------------
# Editorial + Chrome/Metallic Y2K Custom CSS
# ---------------------------------------------------------------------------
# Der CSS-Block ist in zwei Abschnitte gegliedert:
#   A. Minimal-Grundgerüst (ruhiges Editorial-Design)
#   B. Chrome/Y2K-Akzente (nur gezielt bei Upload, Kartenrahmen, Score)
# Alle Farben und Effekte sind zentral definiert und kommentiert.
st.markdown(
    """
    <style>
    /* =====================================================
       RUNWAY LOOK FINDER — Editorial + Chrome/Y2K Accents
       ===================================================== */

    /* -----------------------------------------------------
       A. MINIMAL-GRUNDGERÜST
       Ruhiges, helles Editorial-Design mit viel Weißraum.
       ----------------------------------------------------- */

    /* Google Fonts: Inter für klare, reduzierte Typografie */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* App-Hintergrund: fast reines Weiß, keine durchgängigen Verläufe */
    .stApp {
        background: #FAFAFA;
        color: #1A1A1A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Haupt-Content-Container: hell, luftig, ohne dunkle Glas-Fläche */
    .block-container {
        background: #FFFFFF;
        border-radius: 0;
        box-shadow: none;
        padding: 4rem 3rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Überschriften: groß, ruhig, großzügiger Buchstabenabstand, schwarz/dunkelgrau */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 300;
        letter-spacing: 0.04em;
        color: #111111;
        text-transform: none;
        text-shadow: none;
        margin-bottom: 1.2rem;
    }

    h1 {
        font-size: 3.2rem;
        font-weight: 300;
        letter-spacing: 0.06em;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }

    h2, h3 {
        font-weight: 400;
        font-size: 1.4rem;
        letter-spacing: 0.03em;
        color: #333333;
        margin-top: 3rem;
    }

    /* Fließtext: reduziert, gut lesbar, dunkelgrau */
    p, li, label, small, .stMarkdown p, .stMarkdown div {
        color: #444444;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        line-height: 1.6;
    }

    /* Mehr Weißraum zwischen Abschnitten */
    .element-container {
        margin-bottom: 1.5rem;
    }

    /* Standard-Buttons: zurückhaltend, passend zum Editorial-Look */
    .stButton > button {
        background: #111111;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-transform: none;
        border: none;
        border-radius: 6px;
        padding: 0.7rem 1.6rem;
        box-shadow: none;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #333333;
        transform: translateY(-1px);
    }

    /* -----------------------------------------------------
       B. CHROME / Y2K-AKZENTE
       Nur gezielt bei Upload, Kartenrahmen und Similarity-Score.
       ----------------------------------------------------- */

    /* B1. Upload-Bereich: silbrig-metallischer Verlauf + Glanz */
    .stFileUploader > section {
        background: linear-gradient(135deg, #F5F5F5 0%, #E0E0E0 45%, #C8C8C8 100%);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 12px;
        padding: 2rem;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.9),
            0 4px 16px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }

    .stFileUploader > section:hover {
        border-color: rgba(0, 212, 255, 0.5);
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.95),
            0 0 24px rgba(0, 212, 255, 0.18);
    }

    .stFileUploader small {
        color: #666666;
    }

    /* B2. Ergebnis-Karten: innen hell/weiß, nur der Rahmen als Metallic-Verlauf */
    .result-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0 0.75rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        /* Metallic-Verlauf-Rahmen über border-image */
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 40%, #00D4FF 100%) 1;
        border-image-slice: 1;
        transition: all 0.3s ease;
    }

    .result-card:hover {
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.08);
        transform: translateY(-4px);
    }

    .result-card img {
        border-radius: 10px;
    }

    /* Mehr Abstand zwischen den Ergebnis-Spalten */
    [data-testid="stHorizontalBlock"] > div {
        gap: 1.5rem;
    }

    /* B3. Similarity-Score: Chrome-Blau/Cyan als Akzent */
    .similarity-score {
        display: inline-block;
        color: #00AACC;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-top: 0.6rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.25);
    }

    /* B4. Einmaliger Shine-Sweep über die Ergebnis-Karten beim Laden */
    @keyframes shine-sweep {
        0% { transform: translateX(-120%) skewX(-20deg); }
        100% { transform: translateX(220%) skewX(-20deg); }
    }

    .shine-once {
        position: relative;
    }

    .shine-once::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 40%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.7) 50%,
            transparent 100%
        );
        animation: shine-sweep 1.1s ease-out 1;
        pointer-events: none;
    }

    /* Alert-Boxen zurückhaltend */
    .stAlert {
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
    }

    /* Scrollbar dezent */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #F0F0F0;
    }

    ::-webkit-scrollbar-thumb {
        background: #CCCCCC;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #AAAAAA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 1. Titel und Einführung.
# ---------------------------------------------------------------------------
# st.title zeigt eine große Überschrift an. Alles, was wir mit st.* aufrufen,
# erscheint nacheinander auf der Webseite – in der Reihenfolge, in der es im
# Skript steht.
st.title("Runway Look Finder")
st.markdown(
    "Lade ein Foto hoch und finde die passendsten Looks aus der Gucci-"
    "Kollektion."
)


# ---------------------------------------------------------------------------
# 2. Teure Schritte cachen.
# ---------------------------------------------------------------------------
# Streamlit führt das gesamte Skript bei jeder Interaktion (z. B. Klick,
# Upload, Tastendruck) komplett neu aus. Damit das CLIP-Modell nicht jedes
# Mal neu geladen werden muss, verwenden wir den Decorator @st.cache_resource.
# Er merkt sich das Ergebnis der Funktion und gibt es bei späteren Durchläufen
# wieder zurück.
@st.cache_resource(show_spinner="CLIP-Modell wird geladen …")
def cached_load_clip_model():
    """Lädt das CLIP-Modell einmalig und cached es für die gesamte Sitzung."""
    return load_clip_model(device="cpu")


# Auch das Einlesen von catalog.json lohnt sich zu cachen, weil sich die Datei
# während einer Sitzung nicht ändert. @st.cache_data ist dafür gedacht.
@st.cache_data(show_spinner="Katalog wird geladen …")
def cached_load_catalog(catalog_path="catalog.json"):
    """Lädt den Katalog einmalig und cached ihn für die gesamte Sitzung."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 3. Katalog laden und prüfen.
# ---------------------------------------------------------------------------
CATALOG_FILE = "catalog.json"

if not os.path.exists(CATALOG_FILE):
    # st.error zeigt eine rot hinterlegte Fehlermeldung an. Das Skript läuft
    # danach weiter, daher beenden wir es hier mit st.stop(), damit keine
    # weiteren Schritte ausgeführt werden.
    st.error(
        f"Katalog '{CATALOG_FILE}' nicht gefunden. "
        "Bitte führe zuerst `generate_embedding.py` aus, um Referenz-Looks "
        "zu erstellen."
    )
    st.stop()

catalog = cached_load_catalog(CATALOG_FILE)

if len(catalog) == 0:
    st.error(
        "Der Katalog enthält noch keine Einträge. "
        "Bitte führe zuerst `generate_embedding.py` aus."
    )
    st.stop()


# ---------------------------------------------------------------------------
# 4. Datei-Upload.
# ---------------------------------------------------------------------------
# st.file_uploader zeigt ein Upload-Feld an. Der Rückgabewert ist None, wenn
# noch keine Datei hochgeladen wurde. Ansonsten ist es ein UploadedFile-
# Objekt, das sich wie ein geöffnetes File-Objekt verhält.
uploaded_file = st.file_uploader(
    "Wähle ein Foto aus",
    type=["jpg", "jpeg", "png"],
    help="Unterstützt werden JPG- und PNG-Dateien.",
)

if uploaded_file is None:
    # Noch kein Bild hochgeladen: Hinweis anzeigen und den Rest überspringen.
    st.info("Bitte lade ein Foto hoch, um passende Runway-Looks zu finden.")
    st.stop()


# ---------------------------------------------------------------------------
# 5. Bild anzeigen und Embedding berechnen.
# ---------------------------------------------------------------------------
# Das hochgeladene Bild direkt mit PIL öffnen. Wir müssen es nicht erst auf
# die Festplatte speichern, weil get_image_embedding inzwischen auch PIL-
# Images akzeptiert.
query_image = Image.open(uploaded_file)

st.subheader("Dein hochgeladenes Bild")
st.image(query_image, width="stretch")

# Lade das Modell (aus dem Cache, falls schon geladen).
model, preprocess = cached_load_clip_model()

# Berechne das Embedding für das hochgeladene Bild.
# Wir packen das in einen Spinner, damit der Nutzer sieht, dass gerade
# etwas passiert.
with st.spinner("Embedding wird berechnet und Katalog wird durchsucht …"):
    query_embedding = get_image_embedding(query_image, model, preprocess)
    results = find_similar_images(query_embedding, catalog, top_k=3)


# ---------------------------------------------------------------------------
# 6. Ergebnisse anzeigen.
# ---------------------------------------------------------------------------
st.subheader("Top 3 passende Runway-Looks")

# Ergebnisse in drei Spalten nebeneinander anzeigen. Jede Spalte enthält
# ein Ergebnis. Falls es weniger als 3 Treffer gibt, werden nur so viele
# Spalten erzeugt, wie es Treffer gibt.
cols = st.columns(len(results))

for col, result in zip(cols, results):
    item = result["item"]
    similarity = round(result["similarity"], 3)

    with col:
        # Jeden Treffer in einer "gläsernen" Result-Card verpacken.
        # Der öffnende und schließende Markdown-Call umschließt die
        # Streamlit-Elemente dazwischen.
        st.markdown(
            '<div class="result-card shine-once">',
            unsafe_allow_html=True,
        )

        # Prüfen, ob das Referenzbild überhaupt noch existiert.
        if os.path.exists(item["image_path"]):
            st.image(item["image_path"], width="stretch")
        else:
            st.warning("Bild nicht gefunden")

        # Metadaten zum Treffer anzeigen.
        st.markdown(f"**Designer:** {item['designer']}")
        st.markdown(f"**Kollektion:** {item['collection']}")
        st.markdown(f"**Saison:** {item['season']}")
        st.markdown(f"**Look:** {item['look_number']}")
        st.markdown(
            f'<span class="similarity-score">Ähnlichkeit: {similarity}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
