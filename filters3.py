"""Boat Filter – Gemini + Polars
================================

Versione ottimizzata che usa **Polars** al posto di pandas: su dataset
medio‐grandi (100k‑1M righe) può essere 5‑30× più veloce nei filtri, grazie
al motore eseguito in Rust e alla parallelizzazione automatica.

Flusso:
1. Carichiamo `boats.json` in un `pl.DataFrame`.
2. Gemini converte la query utente in **un’unica espressione Polars** che si
   valuta con `eval()` (es. `df.filter((pl.col('price')<5e5) & (pl.col('max_speed').is_max()))`).
3. Il risultato viene trasformato in elenco puntato con le sole colonne
   citate + nome barca.

Requisiti: python‑dotenv, rich, google‑generativeai, **polars>=0.20**
Chiave API: `GEMINI_API_KEYS` in `.env`.

⚠️ eval() è sempre delicato: sandbox raccomandata in prod.
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import List, Set, Any

import polars as pl
from rich.console import Console
from dotenv import load_dotenv
import google.generativeai as genai

###############################################################################
# 1. Dataset loader                                                          #
###############################################################################

def load_dataset(fp: str | Path = "output_with_contact.json") -> pl.DataFrame:
    data = json.loads(Path(fp).read_text(encoding="utf-8"))
    return pl.DataFrame(data)

###############################################################################
# 2. Prompt template                                                         #
###############################################################################

INSTRUCTION_STR = (
    "1. Convert the query to executable Python code using **Polars** (not pandas).\n"
    "2. The final line must be a Python expression that can be passed to eval().\n"
   
    "3. It **must return a pl.DataFrame** (use head() if necessary).\n"
    "4. PRINT ONLY THE EXPRESSION, no extra text or formatting.\n"
    "5. Do not wrap the expression in quotes or markdown.\n"
    "6. Use `df.filter(<boolean>)`, `pl.lit`, or the lazy API; **avoid** `df[bool_mask]` row‑filter syntax. \n"
    "7.If the query does **not** mention a specific brand, **do not** apply any filter on `boat_name` (or similar brand fields)"

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

###############################################################################
# 3. Gemini helper                                                           #
###############################################################################

def get_polars_expression(query: str, df_sample: str, model: str = "gemini-2.5-flash") -> str:
    prompt = PROMPT_TEMPLATE.format(df_str=df_sample, instructions=INSTRUCTION_STR, query=query)
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(prompt, generation_config={"temperature": 0})
    expr = resp.text.strip()
    if expr.startswith("```"):
        expr = expr.strip("`\n").removeprefix("python").strip()
    return expr

###############################################################################
# 4. Utilities                                                               #
###############################################################################

NAME_COLS = ["boat_name", "Nome della barca", "boatName"]
DEFAULT_EXTRA_COLS = ["price"]



def extract_cols(expr: str) -> List[str]:
    cols: Set[str] = set()
    # pl.col('price') or pl.col("price")
    cols.update(re.findall(r"pl\.col\( ?['\"]([^'\"]+)['\"] ?\)", expr))
    # df['price'] style (rare)
    cols.update(re.findall(r"df\[ ?['\"]([^'\"]+)['\"] ?\]", expr))
    return list(cols)


def bullets(df: pl.DataFrame, cols: List[str]):
    console = Console()
    if df.is_empty():
        console.print("[yellow]Nessun risultato.[/]")
        return

    name_col = next((c for c in NAME_COLS if c in df.columns), None)

    # Colonne da mostrare: richieste + prezzo + (eventuale) name_col
    show_cols = cols.copy()
    for extra in DEFAULT_EXTRA_COLS:
        if extra in df.columns and extra not in show_cols:
            show_cols.append(extra)
    if name_col and name_col not in show_cols:
        show_cols.insert(0, name_col)

    for row in df.rows(named=True):
        title = row.get(name_col, "Barca") if name_col else "Barca"
        console.print(f"\n[bold]• {title}[/]")
        for c in show_cols:
            if c == name_col:
                continue
            if c in row and row[c] is not None:
                console.print(f"   - {c}: {row[c]}")


###############################################################################
# 5. Main loop                                                               #
###############################################################################

def main():
    load_dotenv()
    os.environ['GEMINI_API_KEYS'] = 'AIzaSyBUMXx4ceUhKJanUduKzWrmNauxrYooIIc'

    api_key = os.getenv("GEMINI_API_KEYS")
    if not api_key:
        Console().print("[red]Manca GEMINI_API_KEYS nello .env[/]")
        return
    genai.configure(api_key=api_key)

    df = load_dataset()
    console = Console()
    console.print(f"Dataset caricato: {df.shape[0]} righe")

    df_head_str = df.head().to_pandas().to_string(index=False)  # usiamo pandas solo per stringa

    while True:
        try:
            q = console.input("[bold blue]\nDomanda > [/] ")
        except (KeyboardInterrupt, EOFError):
            break
        q = q.strip()
        if not q:
            continue
        try:
            expr = get_polars_expression(q, df_head_str)
            console.print(f"[cyan]Expr Gemini →[/] {expr}")
            local_ns = {"df": df, "pl": pl}
            res = eval(expr, {}, local_ns)
            if not isinstance(res, pl.DataFrame):
                # se scalar/Series → converti
                res = pl.DataFrame({"result": [res]})
            cols_used = extract_cols(expr)
            bullets(res, cols_used)
        except Exception as e:
            console.print(f"[red]Errore:[/] {e}")
            
            
def query_boats(df: pl.DataFrame, query: str, model: str = "gemini-2.5-flash") -> Tuple[str, pl.DataFrame, List[str]]:
    """Dato un DataFrame e la domanda utente, restituisce:
    - expr: stringa dell'espressione Polars generata da Gemini
    - result_df: DataFrame filtrato (pronto per display)
    - show_cols: ordine colonne da visualizzare (boat_name, price, altre)
    """
    df_head_str = df.head().to_pandas().to_string(index=False)
    expr = get_polars_expression(query, df_head_str, model)

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
    show_cols = list(dict.fromkeys(show_cols))  # unique preserving order
    res =res.head(10)

    return expr, res, show_cols


if __name__ == "__main__":
    main()
