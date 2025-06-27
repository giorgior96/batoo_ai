"""Boat Chat Streamlit Front‑End
================================

Interfaccia chat minimal che si appoggia alla funzione `query_boats`
definita nel modulo back‑end `boat_filter_polars.py`.

Avvio:
    streamlit run boat_chat_streamlit.py

Dipendenze (oltre a quelle del back‑end):
    pip install streamlit polars rich python-dotenv google-generativeai

File richiesti:
* `boat_filter_polars.py` (nel PYTHONPATH o stessa cartella)
* `boats.json` con gli annunci
* `.env` contenente `GEMINI_API_KEYS=<chiave>`
"""

from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
import polars as pl
from dotenv import load_dotenv

# importa back‑end
import filters3 as backend

# ── Config Streamlit ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Batoo AI ✨", page_icon="⛵", layout="wide")

# ── Load dataset ────────────────────────────────────────────────────────────
DATA_PATH = Path("output_with_contact.json")
if not DATA_PATH.exists():
    st.error("⚠️  boats.json non trovato. Caricalo nella cartella o usa lo *file uploader* nella sidebar.")
    st.stop()

df = backend.load_dataset(DATA_PATH)



# ── Chat history state ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]

# ── Title ───────────────────────────────────────────────────────────────────
st.title("Batoo AI ✨")

# initial assistant greeting
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Ciao! Chiedimi pure qualsiasi cosa sulle barche in vendita⛵")
       

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

# ── Chat input ──────────────────────────────────────────────────────────────
if query := st.chat_input("Fai una domanda sulle barche..."):
    # show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # process via backend
    with st.spinner("Batoo sta pensando…"):
      

        try:
            st.write("🔍 Passo 1: chiamata a `query_boats`...")
            expr, results, columns = backend.query_boats(df, query)
            st.write("✅ Espressione ottenuta:", expr)
        except Exception as e:
            answer = f"Si è verificato un errore: {e}"
        else:
            # build bullet answer
            name_col = next((c for c in backend.NAME_COLS if c in results.columns), None)
            lines = []
            if results.is_empty():
                lines.append("Nessun risultato trovato.")
            else:
                for row in results.rows(named=True):
                    lines.append(f"**• {row.get(name_col, 'Barca')}**")
                    for c in columns:
                        if c == name_col or c not in row:
                            continue
                        lines.append(f"&nbsp;&nbsp;- {c}: {row[c]}")
                    lines.append("<br>")
            answer = "\n".join(lines)

    # show assistant answer
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
