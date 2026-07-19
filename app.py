"""
Agentic Research Assistant -- Chat UI (v2: Execution Plan, Citations, Recommendations)

A conversational, chat-style interface for the agentic research paper assistant.
Keeps a running conversation so you can naturally ask follow-ups ("summarize the
first one", "now compare it with the second") the same way you'd chat with a
person -- and shows exactly which tools the Planner Agent decided to run, in order.

Run locally with:
    streamlit run app.py

First run will download the dataset/models and build the embedding cache --
this can take a while on CPU. Subsequent runs reuse the cache.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import pandas as pd
import streamlit as st

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

from transformers import pipeline
from keybert import KeyBERT
import spacy

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq



# Page config + custom styling

st.set_page_config(page_title="Scholar -- Agentic Research Assistant", page_icon="🔭", layout="wide")

st.markdown("""
<style>
    .stChatMessage { border-radius: 14px; }
    div[data-testid="stSidebar"] { border-right: 1px solid #2b2b2b; }
    .tool-badge {
        display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
        border-radius: 999px; font-size: 0.75rem; background: #1f2937; color: #93c5fd;
        border: 1px solid #334155;
    }
    .plan-step {
        display: block; padding: 4px 0; font-size: 0.85rem; color: #a7f3d0;
    }
    .memory-badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.75rem; background: #052e16; color: #4ade80; border: 1px solid #166534;
    }
</style>
""", unsafe_allow_html=True)



# Cached resource loading -- runs once per server session, not per query

@st.cache_resource(show_spinner="Indexing the paper corpus (first run only, this takes a few minutes)...")
def load_resources():
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    df = pd.DataFrame(dataset)[["title", "abstract"]].head(15000).reset_index(drop=True)
    df["paper_text"] = (df["title"] + " " + df["abstract"]).str.replace("\n", " ", regex=False).str.strip()

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("paper_embeddings.npy"):
        embeddings = np.load("paper_embeddings.npy")
    else:
        embeddings = embed_model.encode(df["paper_text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
        np.save("paper_embeddings.npy", embeddings)

    if os.path.exists("paper_faiss.index"):
        index = faiss.read_index("paper_faiss.index")
    else:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "paper_faiss.index")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    kw_model = KeyBERT(embed_model)
    ner_model = spacy.load("en_core_web_sm")

    return df, embed_model, embeddings, index, summarizer, kw_model, ner_model


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


df, embed_model, embeddings, index, summarizer, kw_model, ner_model = load_resources()
llm = load_llm()



# Core helper functions

def raw_search(query, k=5):
    query_embedding = embed_model.encode([query])
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)
    return list(zip(D[0].tolist(), I[0].tolist()))


def raw_summarize(text, max_length=120, min_length=40):
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]


def raw_keywords(text, top_n=5):
    kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=top_n)
    return [k for k, _ in kws]


def raw_entities(text):
    doc = ner_model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]



# LangChain tools

@tool
def search_papers(query: str, k: int = 3) -> str:
    """Search the research paper database for papers relevant to a topic or question.
    Returns title, similarity score, and abstract excerpt for each top-k match."""
    results = raw_search(query, k)
    lines = []
    for score, idx in results:
        row = df.iloc[idx]
        lines.append(
            f"[paper_id={idx}] score={score:.3f} | title=\"{row['title'].strip()}\" | "
            f"abstract_excerpt=\"{row['abstract'][:200].strip()}...\""
        )
    return "\n".join(lines) if lines else "No matching papers found."


@tool
def summarize_paper(paper_id: int) -> str:
    """Generate a concise AI summary of a paper's abstract, given its paper_id."""
    if paper_id < 0 or paper_id >= len(df):
        return f"Invalid paper_id: {paper_id}"
    return raw_summarize(df.iloc[paper_id]["abstract"])


@tool
def extract_keywords(paper_id: int, top_n: int = 5) -> str:
    """Extract the top_n most relevant keywords from a paper's abstract, given its paper_id."""
    if paper_id < 0 or paper_id >= len(df):
        return f"Invalid paper_id: {paper_id}"
    return ", ".join(raw_keywords(df.iloc[paper_id]["abstract"], top_n))


@tool
def extract_entities(paper_id: int) -> str:
    """Extract named entities from a paper's abstract, given its paper_id."""
    if paper_id < 0 or paper_id >= len(df):
        return f"Invalid paper_id: {paper_id}"
    entities = raw_entities(df.iloc[paper_id]["abstract"])
    return ", ".join(f"{t} ({l})" for t, l in entities) if entities else "No named entities found."


@tool
def compare_papers(paper_id_1: int, paper_id_2: int) -> str:
    """Compare two papers side by side, given their paper_ids. Returns a markdown
    table with each paper's title, a short summary, and their cosine similarity."""
    for pid in (paper_id_1, paper_id_2):
        if pid < 0 or pid >= len(df):
            return f"Invalid paper_id: {pid}"
    row1, row2 = df.iloc[paper_id_1], df.iloc[paper_id_2]
    emb1 = embed_model.encode([row1["paper_text"]])
    emb2 = embed_model.encode([row2["paper_text"]])
    faiss.normalize_L2(emb1)
    faiss.normalize_L2(emb2)
    similarity = float(np.dot(emb1[0], emb2[0]))
    summary1 = raw_summarize(row1["abstract"])
    summary2 = raw_summarize(row2["abstract"])

    return (
        "| Aspect | Paper A | Paper B |\n"
        "|---|---|---|\n"
        f"| Title | {row1['title'].strip()} | {row2['title'].strip()} |\n"
        f"| Summary | {summary1} | {summary2} |\n"
        f"| Cosine similarity | {similarity:.3f} (shared value) | {similarity:.3f} (shared value) |\n"
    )


@tool
def recommend_similar_papers(paper_id: int, k: int = 5) -> str:
    """Recommend papers similar to a given paper (not a text query), given its
    paper_id. Use this when the user asks for papers similar to / related to /
    like one they've already found, rather than searching a new topic."""
    if paper_id < 0 or paper_id >= len(df):
        return f"Invalid paper_id: {paper_id}"
    paper_embedding = embeddings[paper_id:paper_id + 1].copy()
    faiss.normalize_L2(paper_embedding)
    D, I = index.search(paper_embedding, k + 1)  # +1 because the paper itself will match
    lines = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == paper_id:
            continue
        row = df.iloc[idx]
        lines.append(f"[paper_id={idx}] score={score:.3f} | title=\"{row['title'].strip()}\"")
        if len(lines) >= k:
            break
    return "\n".join(lines) if lines else "No similar papers found."


@tool
def generate_citation(paper_id: int, style: str = "APA") -> str:
    """Generate a citation for a paper given its paper_id, in the requested style:
    'APA', 'IEEE', or 'BibTeX'. Note: this dataset only contains paper titles and
    abstracts (no author or publication year), so citations will use placeholders
    for missing fields -- always tell the user those fields need to be filled in."""
    if paper_id < 0 or paper_id >= len(df):
        return f"Invalid paper_id: {paper_id}"
    title = df.iloc[paper_id]["title"].strip()
    style = style.strip().lower()

    if style == "ieee":
        citation = f'[Author Unknown], "{title}," arXiv preprint, [Year Unknown].'
    elif style == "bibtex":
        key = "".join(ch for ch in title.split()[0] if ch.isalnum()) or "paper"
        citation = (
            f"@article{{{key}{paper_id},\n"
            f"  title={{ {title} }},\n"
            f"  author={{ Unknown }},\n"
            f"  journal={{arXiv preprint}},\n"
            f"  year={{ Unknown }}\n"
            f"}}"
        )
    else:  # APA default
        citation = f"Author Unknown. (n.d.). {title}. arXiv preprint."

    return (
        f"{citation}\n\n"
        "NOTE: This dataset only provides titles and abstracts, not author names or "
        "publication years, so those fields are placeholders -- fill them in from "
        "the original paper before using this citation."
    )


@tool
def generate_report(paper_id: int, query: str) -> str:
    """Generate a downloadable PDF report for a paper given its paper_id and the
    original user query. Only use when explicitly asked for a report/PDF."""
    if paper_id < 0 or paper_id >= len(df):
        return f"Invalid paper_id: {paper_id}"
    row = df.iloc[paper_id]
    summary = raw_summarize(row["abstract"])
    keywords = raw_keywords(row["abstract"])
    entities = raw_entities(row["abstract"])
    filename = f"Agent_Report_{query[:30].strip().replace(' ', '_')}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename)
    doc.build([
        Paragraph("Agentic Research Assistant -- Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"<b>Query:</b> {query}", styles["Normal"]),
        Paragraph(f"<b>Paper:</b> {row['title'].strip()}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"<b>Summary:</b> {summary}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"<b>Keywords:</b> {', '.join(keywords)}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"<b>Entities:</b> {', '.join(f'{t} ({l})' for t, l in entities) or 'None found'}", styles["Normal"]),
    ])
    # Marker prefix lets the UI detect this result and render a real download button
    return f"REPORT_FILE::{filename}::Report generated: {filename}"


TOOLS = [
    search_papers,
    summarize_paper,
    extract_keywords,
    extract_entities,
    compare_papers,
    recommend_similar_papers,
    generate_citation,
    generate_report,
]

TOOL_LABELS = {
    "search_papers": "Search Papers",
    "summarize_paper": "Summarize",
    "extract_keywords": "Extract Keywords",
    "extract_entities": "Detect Entities",
    "compare_papers": "Compare Papers",
    "recommend_similar_papers": "Recommend Similar Papers",
    "generate_citation": "Generate Citation",
    "generate_report": "Generate PDF Report",
}

SYSTEM_PROMPT = SystemMessage(content="""
You are a Planner Agent for a research paper assistant.

You have access to tools that can search papers, summarize them, extract keywords,
extract entities, compare two papers (as a table), recommend similar papers,
generate citations, and generate a PDF report.

Rules:
1. Only call the tools that are actually needed to answer the user's request.
2. To summarize, extract keywords/entities from, cite, recommend-from, or generate
   a report for a paper, you first need its paper_id -- get this via search_papers
   if you don't already have it from earlier in the conversation.
3. When comparing papers, make sure you have both paper_ids before calling compare_papers.
   Present the compare_papers tool's markdown table output to the user exactly as returned.
4. When the user says "similar to" or "recommend more like" a paper they already found,
   use recommend_similar_papers with that paper's paper_id -- do NOT use search_papers
   with a made-up text query for this.
5. When generating a citation, always pass through the tool's disclaimer about missing
   author/year fields to the user -- do not hide it or invent author names/years yourself.
6. Never invent a paper_id, title, or numeric result -- only use what tools return.
7. After getting tool results, answer the user directly and clearly using that data.
8. Only call generate_report if the user explicitly asks for a report, PDF, or document.
""")


def run_agent(user_input, history, max_tool_iterations=5):
    """Runs the agent loop and returns (final_answer, tool_trace, updated_history)."""
    llm_with_tools = llm.bind_tools(TOOLS)
    history = history + [HumanMessage(content=user_input)]
    trace = []

    for _ in range(max_tool_iterations):
        response = llm_with_tools.invoke(history)
        history = history + [response]

        if not response.tool_calls:
            return response.content, trace, history

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            matching_tool = next((t for t in TOOLS if t.name == tool_name), None)

            if matching_tool is None:
                tool_result = f"Error: no such tool '{tool_name}'"
            else:
                try:
                    tool_result = matching_tool.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Tool error: {e}"

            tool_result_str = str(tool_result)
            trace.append({"tool": tool_name, "args": tool_args, "result": tool_result_str})

            # The LLM only needs the human-readable part -- strip our internal marker
            llm_visible_result = tool_result_str
            if llm_visible_result.startswith("REPORT_FILE::"):
                llm_visible_result = llm_visible_result.split("::", 2)[-1]

            history = history + [ToolMessage(content=llm_visible_result, tool_call_id=tool_call["id"])]

    return "Reached max tool iterations without a final answer.", trace, history


# Sidebar

with st.sidebar:
    st.markdown("### 🔭 Scholar")
    st.caption("An agentic assistant over a corpus of ML research papers.")

    st.markdown('<span class="memory-badge">Memory: ON</span>', unsafe_allow_html=True)
    st.caption("The agent remembers papers, paper_ids, and results from earlier in this conversation.")

    st.divider()
    st.markdown("**Available tools**")
    tool_names = [t.name for t in TOOLS]
    st.markdown("".join(f'<span class="tool-badge">{TOOL_LABELS.get(n, n)}</span>' for n in tool_names), unsafe_allow_html=True)

    st.divider()
    st.markdown("**How it works**")
    st.caption(
        "Every message goes to a Planner Agent (Groq Llama 3.1). It reads your "
        "request, decides which tool(s) it actually needs, calls them in order, "
        "and answers using only what the tools return."
    )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = [SYSTEM_PROMPT]
        st.rerun()



# Main chat area

st.markdown("## Ask Scholar about a research topic")
st.caption(
    "e.g. \"Find papers on vision transformers\" -> \"compare the first and third\" "
    "-> \"recommend more like the first\" -> \"cite it in APA\" -> \"generate a PDF report\""
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = [SYSTEM_PROMPT]


def render_execution_plan(trace):
    """Render the sequence of tools the agent actually executed, checklist-style --
    this is the Planner Agent's plan made visible."""
    if not trace:
        st.caption("No tools were needed -- the agent answered directly.")
        return
    st.markdown("**Execution Plan**")
    for step in trace:
        label = TOOL_LABELS.get(step["tool"], step["tool"])
        st.markdown(f'<span class="plan-step">Done: {label}</span>', unsafe_allow_html=True)


def render_trace_detail(trace):
    for step in trace:
        st.markdown(f"**`{step['tool']}`**  \nargs: `{step['args']}`")
        result = step["result"]
        if result.startswith("REPORT_FILE::"):
            _, filename, _ = result.split("::", 2)
            render_download_button(filename)
        elif "|---|" in result or result.strip().startswith("|"):
            st.markdown(result)  # render markdown tables natively
        else:
            st.code(result[:500] + ("..." if len(result) > 500 else ""), language=None)
        st.divider()


def render_download_button(filename):
    """Show a real download button for a generated PDF, if the file exists on disk."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            st.download_button(
                label=f"Download {filename}",
                data=f.read(),
                file_name=filename,
                mime="application/pdf",
                key=f"dl_{filename}_{id(f)}",
            )
    else:
        st.warning(f"Report file not found on disk: {filename}")


# Replay past turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            render_execution_plan(msg["trace"])
            with st.expander(f"{len(msg['trace'])} tool call(s) -- full detail"):
                render_trace_detail(msg["trace"])

user_input = st.chat_input("Ask about research papers...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Planning and calling tools..."):
            answer, trace, updated_history = run_agent(user_input, st.session_state.history)
            st.session_state.history = updated_history
        st.markdown(answer)
        render_execution_plan(trace)
        if trace:
            with st.expander(f"{len(trace)} tool call(s) -- full detail"):
                render_trace_detail(trace)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})