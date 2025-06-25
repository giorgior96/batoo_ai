import os
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict

import google.generativeai as genai

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_google_genai import GoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages


import re

NUMERIC_FIELDS = {
    "lunghezza": "length",
    "long": "length",
    "length": "length",
    "larghezza": "width",
    "price": "price",
    "prezzo": "price",
    "anno": "year",
    "year": "year",
    "velocità massima": "max_speed",
    "max speed": "max_speed",
}

OPS_MAP = {">=": "$gte", "<=": "$lte", ">": "$gt", "<": "$lt"}

NUMERIC_PATTERN = re.compile(
    r"(?P<label>\b(?:lunghezza|length|larghezza|price|prezzo|anno|year|velocità massima|max speed)\b)"
    r"\s*(?P<op>>=|<=|>|<|\bpiù di\b|\bmeno di\b|\bsopra\b|\bsotto\b)?"
    r"\s*(?P<value>\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)

TEXT_PATTERN = re.compile(
    r"cantiere\s+(?P<yard>[A-Za-z0-9]+)|motore\s+(?P<engine>[A-Za-z0-9]+)",
    re.IGNORECASE,
)

def extract_filters(question: str) -> dict:
    """
    Estrae un dizionario di filtri Chroma dal testo utente.
    Supporta comparatori numerici e match testuali essenziali.
    """
    filters = {}

    # --- numerici ---
    for m in NUMERIC_PATTERN.finditer(question):
        raw_label = m.group("label").lower()
        op = m.group("op") or ">="  # default a >= se non specificato
        op = op.strip().lower()
        # normalizza "più di", "sopra", "over" → ">"
        if op in ("più di", "sopra", "over"):
            op = ">"
        elif op in ("meno di", "sotto", "under"):
            op = "<"
        value = float(m.group("value").replace(",", "."))
        field = NUMERIC_FIELDS[raw_label]
        filters.setdefault(field, {})[OPS_MAP[op]] = value

    for token in re.findall(r"\w+", question.lower()):
        if token in BRAND_VOCAB:
            filters["boat_name"] = {"$in": [token]}
            break

    return filters



load_dotenv()

os.environ['GEMINI_API_KEYS'] = 'AIzaSyBUMXx4ceUhKJanUduKzWrmNauxrYooIIc'


# === Paths & Embedding ===
CHROMA_PATH = "chroma"
gpt4all_embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=gpt4all_embeddings)

# === Gemini model ===
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    google_api_key=os.environ['GEMINI_API_KEYS']
)


def build_brand_set():
    brand_tokens = set()
    for md in db.get()["metadatas"]:
        if not md or "boat_name" not in md:
            continue
        for word in md["boat_name"].split():
            w = word.lower()
            if len(w) > 2:        # scarta articoli / preposizioni
                brand_tokens.add(w)
    return brand_tokens

BRAND_VOCAB = build_brand_set()  


# === RETRIEVER DINAMICO ===
def get_retriever(k: int = 20, filters: dict | None = None):
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={
    "k": k,
    **({"filter": filters} if filters else {})  # ← SOLO SE filters esiste
}
    )


# === CONTEXTUALIZER ===
def contextualize_question(k: int = 20, filters: dict | None = None):
    question_reformulation_prompt = """
    Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    question_reformulation_template = ChatPromptTemplate.from_messages(
        [
            ("system", question_reformulation_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever = get_retriever(k=k, filters=filters)

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        question_reformulation_template,
    )
    
    return history_aware_retriever


# === QA CHAIN ===
def answer_question(k: int = 20, filters: dict | None = None):
    answer_question_prompt = """ 
    Usa i seguenti dati tecnici per rispondere in modo preciso alla domanda dell’utente.
    Ignora le barche con "prezzo su quotazione" o con prezzo non numerico se l’utente chiede un range di prezzo.
    Cerca di usare tutti i valori numerici (es. lunghezza, larghezza, velocità) e testuali (motore, colore, accessori).
    Non inventare. Se non trovi il dato, rispondi 'Non disponibile'. 
    Se l'utente chiede il nome di una barca cerca per boat name dai il prezzo corretto.

    {context}"""
    
    answer_question_template = ChatPromptTemplate.from_messages(
        [
            ("system", answer_question_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 🔁 Puoi sostituire con create_refine_documents_chain se superi i token
    answer_question_chain = create_stuff_documents_chain(llm, answer_question_template)

    history_aware_retriever = contextualize_question(k=k, filters=filters)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_question_chain)
    
    return rag_chain


# === STATO DELLA CONVERSAZIONE ===
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


# === NODO MODELLO ===
def call_model(state: State):
    user_question = state["input"]
    filters = extract_filters(user_question)
    
    # 2️⃣ Costruisci la chain con quei filtri
    rag_chain = answer_question(k=20, filters=filters if filters else None)

    response = rag_chain.invoke(state)

    print(f"[DEBUG] Docs ctx len={len(response['context'])}, filters={filters}")

    return {
        "chat_history": [
            HumanMessage(user_question),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

# === LANGGRAPH WORKFLOW ===
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# === FUNZIONE PRINCIPALE ===
def execute_user_query(query_text: str) -> str:
    config = {"configurable": {"thread_id": "abc123"}}

    result = app.invoke(
        {"input": query_text},
        config=config,
    )
    
    return result["answer"]


# === TEST LOCALE ===
if __name__ == "__main__":
    execute_user_query(query_text)
    print(result)
