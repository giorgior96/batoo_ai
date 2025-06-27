from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
import polars as pl
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
from typing import List, Set, Tuple

# ───── Config ─────
st.set_page_config(page_title="Batoo AI ✨", page_icon="⛵", layout="wide")
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEYS"))

# ───── Costanti ─────
NAME_COLS = ["boat_name", "Nome della barca", "boatName"]
DEFAULT_EXTRA_COLS = ["price"]

INSTRUCTION_STR = (
    "1. Convert the query to executable Python code using **Polars** (not pandas).\n"
    "2. The final line must be a Python expression that can be passed to eval().\n"
    "3. It **must return a pl.DataFrame** (use head() if necessary).\n"
    "4. PRINT ONLY THE EXPRESSION, no extra text or formatting.\n"
    "5. Do not wrap the expression in quotes or markdown.\n"
    "6. Use `df.filter(<boolean>)`, `pl.lit`, or the lazy API.\n"
    "7. If no brand is mentioned, do NOT filter on `boat_name` or similar."
)

PROMPT_TEMPLATE = (
    "You are working with a Polars DataFrame in Python.\n"
    "The DataFrame variable is named `df`.\n"
    "Here is the output of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions strictly:\n"
    "{instructions}\n"
    "Query: {query}\n\n"
    "Expression:"
)

# ───── Funzioni ─────
def load_dataset(fp: str | Path = "output_with_contact.json") -> pl.DataFrame:
    data = json.loads(Path(fp).read_text(encoding="utf-8"))
    return pl.DataFrame(data)

def get_polars_expression(query: str, df_sample: str, model: str = "gemini-1.5-flash-latest") -> str:
    prompt = PROMPT_TEMPLATE.format(df_str=df_sample, instructions=INSTRUCTION_STR, query=query)
    st.write("🧠 Prompt creato. Invio richiesta a Gemini...")
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(prompt, generation_config={"temperature": 0})
    st.write("✅ Risposta ricevuta da Gemini")
    expr = resp.text.strip()
    if expr.startswith("```"):
        expr = expr.strip("`\n").removeprefix("python").strip()
    return expr

def extract_cols(expr: str) -> List[str]:
    cols: Set[str] = set()
    cols.update(re.findall(r"pl\\.col\\( ?['\"]([^'\"]+)['\"] ?\\)", expr))
    cols.update(re.findall(r"df\\[ ?['\"]([^'\"]+)['\"] ?\\]", expr))
    return list(cols)

def query_boats(df: pl.DataFrame, query: str, model: str = "gemini-1.5-flash-latest") -> Tuple[str, pl.DataFrame, List[str]]:
    df_head_str = df.head().to_pandas().to_string(index=False)
    expr = get_polars_expression(query, df_head_str, model)
    st.code(expr, language="python")
    local_ns = {"df": df, "pl": pl}
    res = eval(expr, {}, local_ns)
    if isinstance(res, pl.LazyFrame):
        res = res.collect()
    if not isinstance(res, pl.DataFrame):
        res = pl.DataFrame({"result": [res]})
    cols_used = extract_cols(expr)
    name_col = next((c for c in NAME_COLS if c in df.columns), None)
    show_cols = cols_used + [c for c in DEFAULT_EXTRA_COLS if c in df.columns]
    if name_col and name_col not in show_cols:
        show_cols.insert(0, name_col)
    show_cols = list(dict.fromkeys(show_cols))
    return expr, res.head(10), show_cols

# ───── Avvio app ─────
st.title("Batoo AI ✨")

DATA_PATH = Path("output_with_contact.json")
if not DATA_PATH.exists():
    st.error("❌ File output_with_contact.json mancante.")
    st.stop()

df = load_dataset(DATA_PATH)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Ciao! Chiedimi pure qualsiasi cosa sulle barche in vendita ⛵")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

if query := st.chat_input("Fai una domanda sulle barche..."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Batoo sta pensando…"):
        try:
            expr, results, columns = query_boats(df, query)
            name_col = next((c for c in NAME_COLS if c in results.columns), None)
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
        except Exception as e:
            answer = f"❌ Errore: `{e}`"
            st.error(answer)

    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
