"""
clip_utils.py
=============
Gemeinsame Hilfsfunktionen für den Umgang mit CLIP in diesem Projekt.
Diese Datei wird von generate_embedding.py und find_similar.py genutzt,
damit das Modell nicht an mehreren Stellen geladen werden muss.
"""

import numpy as np
import torch
from PIL import Image
import open_clip


# Standard-Konfiguration für das CLIP-Modell.
# Beide Skripte verwenden dieselbe Architektur und dieselben Gewichte,
# damit die Embeddings vergleichbar sind.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


def load_clip_model(device="cpu"):
    """
    Lädt das CLIP-Modell und die passende Bildvorverarbeitung.

    Args:
        device: "cpu" oder "cuda". Standard ist "cpu", da es überall läuft.

    Returns:
        Ein Tupel (model, preprocess). model ist das CLIP-Modell,
        preprocess ist die Funktion, die ein PIL-Bild in einen Tensor
        für das Modell umwandelt.
    """
    # open_clip.create_model_and_transforms lädt in einem Schritt:
    #   - das eigentliche CLIP-Modell
    #   - die passende Vorverarbeitung (Bild skalieren, zentrieren, in Tensor)
    #   - den Tokenizer für Text (hier nicht nötig, daher "_")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )

    # In den Evaluierungsmodus schalten: Dropout etc. werden deaktiviert.
    model.eval()

    return model, preprocess


def get_image_embedding(image_input, model, preprocess):
    """
    Berechnet das normalisierte CLIP-Embedding für ein Bild.

    Args:
        image_input: Entweder ein Pfad zur Bilddatei (str) oder ein bereits
            geöffnetes PIL-Image. Das ist praktisch für Streamlit, weil
            st.file_uploader ein Datei-Objekt liefert, das direkt in ein
            PIL-Image umgewandelt werden kann.
        model: Das geladene CLIP-Modell.
        preprocess: Die Vorverarbeitungsfunktion aus load_clip_model().

    Returns:
        Ein NumPy-Array der Form (1, 512) mit dem normalisierten Embedding.
    """
    # Falls ein Pfad übergeben wurde, öffnen wir das Bild mit PIL.
    # Ansonsten nehmen wir an, dass es sich bereits um ein PIL-Image handelt.
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    # preprocess wandelt das Bild in einen Tensor um.
    # .unsqueeze(0) fügt eine Batch-Dimension hinzu:
    # [3, 224, 224] -> [1, 3, 224, 224].
    # CLIP erwartet immer einen Batch, auch wenn es nur ein Bild ist.
    image_tensor = preprocess(image).unsqueeze(0)

    # torch.no_grad() spart Speicher, da keine Gradienten berechnet werden
    # (wir machen nur eine Vorhersage, kein Training).
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    # CLIP-Embeddings werden auf Länge 1 normalisiert. Das ist wichtig,
    # damit die Kosinus-Ähnlichkeit später einfach dem Skalarprodukt
    # entspricht (Wertebereich -1 bis 1).
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    # Vom PyTorch-Tensor zu einem NumPy-Array konvertieren.
    return embedding.numpy().astype(np.float32)
