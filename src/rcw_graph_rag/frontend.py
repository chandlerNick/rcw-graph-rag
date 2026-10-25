import os
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

from rcw_graph_rag import RCWGraphRAG


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT_DIR / "rcw_db"
MAX_CONTEXT_CHARS = 50000

st.set_page_config(page_title="WA Law GraphRAG Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ WA Law GraphRAG Assistant")
st.caption(
    "An experimental research aid. Verify legal information against official sources; "
    "this tool does not provide legal advice."
)

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
)
model_choice = st.sidebar.selectbox(
    "Select Gemini Model",
    ["gemini-3.5-flash", "gemini-3.8-flash"],
)
db_dir = Path(os.environ.get("RCW_DB_DIR", str(DEFAULT_DB_DIR))).expanduser()
top_k = st.sidebar.slider("Vector matches", min_value=1, max_value=10, value=5)
graph_hops = st.sidebar.slider("Graph hops", min_value=0, max_value=2, value=1)
max_related = st.sidebar.slider(
    "Maximum linked sections", min_value=0, max_value=10, value=5
)
st.sidebar.caption(
    "Your question and retrieved RCW text are sent to Google Gemini to generate an answer."
)

if not api_key:
    st.info(
        "Enter a Google Gemini API key in the sidebar or set GEMINI_API_KEY to continue.",
        icon="🔑",
    )
    st.stop()


@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)


@st.cache_resource
def get_rag(database_dir):
    return RCWGraphRAG(database_dir)


def make_context(result):
    sources = [
        {**record, "source_type": "Vector match"}
        for record in result["hits"]
    ] + [
        {**record, "source_type": f"Graph-linked via {record['via']}"}
        for record in result["graph_context"]
    ]

    blocks = []
    used_sources = []
    remaining = MAX_CONTEXT_CHARS
    truncated = False
    for source in sources:
        url = source.get("url", "")
        block = (
            f"[{source['id']}] {source['section_title']}\n"
            f"Title {source['title_number']}, Chapter {source['chapter_number']}\n"
            f"Source URL: {url}\n"
            f"Retrieval: {source['source_type']}\n"
            f"Text:\n{source['text']}"
        )
        separator = "\n\n"
        available = remaining - len(separator)
        if available <= 0:
            truncated = True
            break
        if len(block) > available:
            block = block[:available]
            truncated = True
        blocks.append(block)
        used_sources.append(source)
        remaining -= len(block) + len(separator)
        if truncated:
            break

    context_text = "\n\n".join(blocks)
    if truncated:
        context_text += "\n\n[Retrieved context was truncated to fit the request.]"
    return context_text, used_sources, truncated


def format_conversation(messages):
    recent_messages = messages[-8:]
    return "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: "
        f"{message['content']}"
        for message in recent_messages
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Washington State Law..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching RCW sections and graph connections..."):
                rag = get_rag(str(db_dir))
                retrieved = rag.retrieve(
                    prompt,
                    k=top_k,
                    hops=graph_hops,
                    max_extra=max_related,
                )
                context, sources, truncated = make_context(retrieved)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not load or query the RCW database at {db_dir}: {error}")
            st.stop()

        if not sources:
            st.warning("No RCW sections were retrieved for this question.")

        conversation = format_conversation(st.session_state.messages[:-1])
        augmented_prompt = (
            f"Conversation so far:\n{conversation or '(none)'}\n\n"
            f"Current question:\n{prompt}\n\n"
            f"Retrieved RCW source material:\n"
            f"{context or '(No source material was retrieved.)'}\n\n"
            "Structure your response in this order:\n"
            "1. Start with a short **Summary** (1–3 sentences) that gives the main "
            "takeaway and synthesizes the retrieved source material.\n"
            "2. Follow with **Answer**, providing the fuller explanation and relevant "
            "details.\n"
            "Cite every legal claim in both sections with the exact RCW section "
            "identifier in square brackets, such as [RCW 9A.04.110]. Only cite sections "
            "included above. Include a Markdown link to the source URL provided for each "
            "cited section when available. If the retrieved text does not answer the "
            "question, say what is missing and do not guess. Distinguish the text of "
            "the statute from any explanation, and remind the user to verify the current "
            "official law."
        )

        response_placeholder = st.empty()
        full_response = ""
        try:
            config = types.GenerateContentConfig(
                system_instruction=(
                    "You are an informational assistant for Washington State law. "
                    "You are not a lawyer and do not provide legal advice. Be precise, "
                    "ground answers in the supplied RCW text, acknowledge uncertainty, "
                    "and never invent legal citations."
                ),
                temperature=0.2,
            )
            response_stream = get_client(api_key).models.generate_content_stream(
                model=model_choice,
                contents=augmented_prompt,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(
                full_response or "The model returned an empty response."
            )
        except genai.errors.APIError as error:
            st.error(f"Gemini API error: {error}")

        if truncated:
            st.warning("The retrieved context was shortened before being sent to Gemini.")

        with st.expander(f"Sources retrieved ({len(sources)})", expanded=True):
            if not sources:
                st.write("No source sections were found.")
            for source in sources:
                title = f"{source['id']} — {source['section_title']}"
                st.markdown(f"**{title}**  \n{source['source_type']}")
                if source.get("url"):
                    st.link_button("Open source section", source["url"])
                st.write(source.get("text", ""))

        if full_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
