import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
import glob
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import shutil
from langchain_chroma import Chroma


### Set variables ###
DATA_PATH = "data_sources"

CHROMA_PATH = "chroma"

gpt4all_embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)


def create_vector_db():
    "Create vector DB from personal PDF files."
    documents = load_documents()
    doc_chunks = split_text(documents)
    save_to_chroma(doc_chunks)

def load_documents():
    path = os.path.join("data_sources", "output_with_contact.json")  # ← usa il tuo path reale
    with open(path, encoding="utf-8") as f:
        boats = json.load(f)  # Lista di dizionari

    numeric_fields = {"length", "width", "price", "seats", "year", "max_speed", "cruising_speed", "draft"}

    docs = []
    for boat in boats:
        # Escludi immagini
        excluded_keys = {"image_urls"}

        # Prepara i metadata puliti
        metadata = {}
        for k, v in boat.items():
            if k in excluded_keys:
                continue
            if k in numeric_fields:
                try:
                    metadata[k] = float(v)
                except:
                    pass
            elif isinstance(v, (str, int, float, bool)):
                metadata[k] = v

        # Prepara il testo da embeddare
        text = "\n".join(
            f"{k}: {v}" for k, v in boat.items() if k not in excluded_keys
        )

        docs.append(Document(page_content=text, metadata=metadata))

    print(f"✅ Indicizzate {len(docs)} barche nel dataset.")
    return docs

def split_text(documents: list[Document]):
    "Split documents into chunks."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=0
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    "Clear previous db, and save the new db."
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # create db
    db = Chroma.from_documents(
        chunks, gpt4all_embeddings, persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
if __name__ == "__main__":    
    create_vector_db()