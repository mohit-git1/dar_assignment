import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution context
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DISTANCE_THRESHOLD,
    DISTANCE_THRESHOLD_SCOPED,
    MODEL,
)
from app.rag import generate, get_collection

st.set_page_config(
    page_title="DarGlobal & Wasalt AI Property Assistant",
    page_icon="🏢",
    layout="wide",
)

# Custom header
st.title("🏢 DarGlobal & Wasalt AI Property Assistant")
st.caption("A RAG-powered chatbot answering questions over verified public property data.")

# Sidebar Controls
with st.sidebar:
    st.header("Search & Data Settings")

    source_display = st.radio(
        "Filter by Source:",
        options=["All", "DarGlobal", "Wasalt"],
        index=0,
        help="Filter property retrieval to all sources or a specific platform.",
    )
    source_filter_map = {"All": "all", "DarGlobal": "darglobal", "Wasalt": "wasalt"}
    selected_source = source_filter_map[source_display]

    st.divider()
    st.subheader("Collection Stats")
    try:
        col = get_collection()
        total_chunks = col.count()
        st.metric("Total Indexed Chunks", total_chunks)
    except Exception:
        total_chunks = 0
        st.metric("Total Indexed Chunks", "0")

    st.caption(f"**Embedder:** Chroma Default (MiniLM-L6-v2 ONNX)")
    st.caption(f"**Distance Metric:** Cosine (Unscoped: {DISTANCE_THRESHOLD} / Scoped: {DISTANCE_THRESHOLD_SCOPED})")
    st.caption(f"**LLM Model:** {MODEL}")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! I am your AI Property Assistant for **DarGlobal** and **Wasalt**.\n\n"
                "You can ask me about available villas, apartments for rent or sale, locations, "
                "or luxury projects. Choose a suggested question below or type your own!"
            ),
            "sources": [],
        }
    ]

# Suggested Questions
st.markdown("##### 💡 Suggested Questions")
suggested_questions = [
    "What villas are for sale in Riyadh?",
    "What apartments are available for rent in Jeddah?",
    "What properties are for sale in Makkah?",
    "What are the luxury projects by DarGlobal?",
    "What residential options exist in Al-Arid or An-Narjis?",
]

cols = st.columns(len(suggested_questions))
clicked_suggestion = None
for i, col in enumerate(cols):
    with col:
        if st.button(suggested_questions[i], key=f"sugg_{i}", use_container_width=True):
            clicked_suggestion = suggested_questions[i]

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("---")
            st.markdown("**Verified Sources:**")
            for url in msg["sources"]:
                st.markdown(f"- [{url}]({url})")

# Determine active prompt (from chat input or suggested button)
user_input = st.chat_input("Ask a question about properties, prices, or locations...")
active_prompt = clicked_suggestion or user_input

if active_prompt:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching verified property data..."):
            result = generate(
                query=active_prompt,
                history=st.session_state.messages[:-1],
                source_filter=selected_source,
            )
            answer_text = result["answer"]
            sources = result["sources"]

            st.markdown(answer_text)

            if sources:
                st.markdown("---")
                st.markdown("**Verified Sources:**")
                for url in sources:
                    st.markdown(f"- [{url}]({url})")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
    })
