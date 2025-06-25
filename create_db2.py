import os, json, glob, shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain.schema import Document
from tqdm import tqdm   # solo per progress bar

from langchain_google_genai import GoogleGenerativeAIEmbeddings


os.environ['GEMINI_API_KEYS'] = 'AIzaSyBUMXx4ceUhKJanUduKzWrmNauxrYooIIc'



DATA_JSON = Path("data_sources/output_with_contact.json")       # unico file
CHROMA_DIR = Path("chroma")

EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GEMINI_API_KEYS"]
)
NUMERIC_KEYS = {
    "length", "width", "price", "seats", "year",
    "max_speed", "cruising_speed", "draft", "weight",
    "water_tank", "fuel", "engine_power"
}

EXCLUDE_KEYS = {"image_urls"}

def clean_numeric(value):
    """Converte 'EUR 79.500,-' -> 79500.0, oppure ritorna None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # togli puntini mille e simboli
    v = (
        str(value)
        .replace("EUR", "")
        .replace("€", "")
        .replace(".", "")
        .replace(",", ".")
        .strip()
    )
    try:
        return float(v)
    except ValueError:
        return None

def load_documents() -> list[Document]:
    boats = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    docs: list[Document] = []

    for boat in tqdm(boats, desc="Indicizzo barche"):
        # ─── nome/brand/modello normalizzati ───
        boat_name  = boat.get("boat_name", "")
        brand      = boat_name.split()[0].lower() if boat_name else ""
        model      = " ".join(boat_name.split()[1:]).lower() if boat_name else ""
        category = boat.get("category")


        # ─── metadati puliti ───
        metadata = {
            "boat_name": boat_name,
            "brand_normalized": brand,
            "model_normalized": model,
            "country": boat.get("country"),
            "region": boat.get("region"),
            "location": boat.get("location"),
            "category": ", ".join(category) if isinstance(category, list) else category,
            "color": boat.get("color"),
            "engine": boat.get("engine"),
            "propulsion": boat.get("propulsion"),
            "condition": boat.get("condition"),
        }

        # aggiungi numerici castati
        for k in NUMERIC_KEYS:
            num = clean_numeric(boat.get(k))
            if num is not None:
                metadata[k] = num

        # ─── testo embeddabile ───
        text_lines = [boat_name]
        for k, v in boat.items():
            if k in EXCLUDE_KEYS:
                continue
            if isinstance(v, list):
                v = ", ".join(map(str, v))
            text_lines.append(f"{k}: {v}")
        page_text = "\n".join(text_lines)

        docs.append(Document(page_content=page_text, metadata=metadata))

    return docs

def build_chroma():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    docs = load_documents()
    print(f"✅ Indicizzate {len(docs)} barche")
    Chroma.from_documents(docs, EMBEDDINGS, persist_directory=str(CHROMA_DIR))
    print("✅ Vector-DB salvato:", CHROMA_DIR)

if __name__ == "__main__":
    build_chroma()
