# === rag_server.py (aggiornato) =====================================================
"""
Server RAG con:
• brand_normalized nei metadati
• filtri numerici + brand automatici
• k dinamico (150 generico, 40 con filtri)
• esclude boat senza prezzo numerico quando necessario
"""

import os, re
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)


from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ╭─────────────────────── COSTANTI REGEX & MAP ───────────────────────╮
NUMERIC_FIELDS = {
    "lunghezza": "length", "length": "length",
    "larghezza": "width",
    "price": "price", "prezzo": "price",
    "anno": "year", "year": "year",
    "velocità massima": "max_speed", "max speed": "max_speed",
}
OPS_MAP = {">=": "$gte", "<=": "$lte", ">": "$gt", "<": "$lt"}
NUMERIC_PATTERN = re.compile(
    r"(?P<label>\\b(?:lunghezza|length|larghezza|price|prezzo|anno|year|velocità massima|max speed)\\b)"
    r"\\s*(?P<op>>=|<=|>|<|\\bpiù di\\b|\\bmeno di\\b|\\bsopra\\b|\\bsotto\\b)?"
    r"\\s*(?P<value>\\d+(?:[\\.,]\\d+)?)", re.IGNORECASE)
# ╰────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── ENV & VECTOR DB ─────────────────────────╮
load_dotenv()
CHROMA_PATH = "chroma"
os.environ['GEMINI_API_KEYS'] = 'AIzaSyBUMXx4ceUhKJanUduKzWrmNauxrYooIIc'


embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={"allow_download": "True"},
)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# costruisci set brand da campo brand_normalized
BRAND_VOCAB = {
    (md.get("boat_name") or "").lower()
    for md in db.get()["metadatas"] if md and md.get("brand_normalized")
}

# ╰────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── LLM (Gemini) ──────────────────────────╮
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    google_api_key=os.environ["GEMINI_API_KEYS"],
    temperature=0.2,
)
# ╰────────────────────────────────────────────────────────────────────╯

# ╭────────────────────── FUNCTION: extract_filters ───────────────────╮

def extract_filters(question: str) -> dict:
    filters: dict = {}

    # numerici
    for m in NUMERIC_PATTERN.finditer(question):
        field = NUMERIC_FIELDS[m.group("label").lower()]
        op_txt = (m.group("op") or ">=").strip().lower()
        op_txt = ">" if op_txt in ("più di", "sopra", "over") else "<" if op_txt in ("meno di", "sotto", "under") else op_txt
        value = float(m.group("value").replace(",", "."))
        filters.setdefault(field, {})[OPS_MAP[op_txt]] = value

    # brand automatico
    for token in re.findall(r"\w+", question.lower()):
        if token in BRAND_VOCAB:
            filters["boat_name"] = {"$eq": token}
            break

    return filters

# ╰────────────────────────────────────────────────────────────────────╯

# ╭──────────────────── RETRIEVER & CONTEXTUALIZER ────────────────────╮

def get_retriever(filters: dict | None):
    k = 150 if not filters else 40
    kwargs = {"k": k}
    if filters:
        kwargs["filter"] = filters
    return db.as_retriever(search_type="similarity", search_kwargs=kwargs)


def contextualize_question(filters: dict | None):
    tmpl = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the latest user question as a standalone query."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(llm, get_retriever(filters), tmpl)

# ╰────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── QA CHAIN  ───────────────────────────────╮

def answer_question(filters: dict | None):
    prompt = (
        "Usa solo i dati presenti. Se il prezzo è mancante o su quotazione, "
        "scrivi 'Prezzo non disponibile'.\n\n{context}")
    qa_tmpl = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_tmpl)
    return create_retrieval_chain(contextualize_question(filters), qa_chain)

# ╰────────────────────────────────────────────────────────────────────╯

# ╭───────────────────────── LANGGRAPH NODO ───────────────────────────╮
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def call_model(state: State):
    q = state["input"]
    filters = extract_filters(q)
    rag = answer_question(filters or None)
    resp = rag.invoke(state)
    print(f"[DEBUG] ctx_len={len(resp['context'])}  filters={filters}")
    return {
        "chat_history": [HumanMessage(q), AIMessage(resp["answer"])],
        "context": resp["context"],
        "answer": resp["answer"],
    }

workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

# ╭────────────────────────── PUBLIC API ──────────────────────────────╮

def execute_user_query(query: str) -> str:
    return app.invoke({"input": query}, config={"configurable": {"thread_id": "1"}})["answer"]

if __name__ == "__main__":
     execute_user_query(query_text)
