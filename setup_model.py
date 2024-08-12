import spacy
from spacy.cli import download

# Function to ensure the model is downloaded
def ensure_model():
    try:
        spacy.load("en_core_web_md")
    except OSError:
        print("Model not found. Downloading...")
        download("en_core_web_md")

# Ensure the model is downloaded
ensure_model()

