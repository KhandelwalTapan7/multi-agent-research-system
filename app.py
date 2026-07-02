"""
Multi-Agent Research System — Streamlit UI
============================================================
A polished, feature-rich front-end for the LangChain + Mistral
powered multi-agent research pipeline:

    Search Agent  ->  Reader Agent  ->  Writer Chain  ->  Critic Chain

Features
--------
- Live per-agent status badges + progress bar while the pipeline runs
- Metrics dashboard (word counts, critic score, error count)
- Tabbed results view (Report / Critic Feedback / Search / Scraped)
- Run history with reload, compare, and clear
- Light / Dark theme toggle
- JSON + Markdown export of results
- Quick topic suggestion chips
- Sidebar analytics (total runs, average duration)
- Defensive error handling around every agent/chain call

Run with:
    streamlit run app.py
"""

import json
import time
import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


# ============================================================
# CONSTANTS
# ============================================================
APP_TITLE = "Multi-Agent Research System"
APP_ICON = "🔎"
MAX_HISTORY_ITEMS = 15

# Retry/backoff settings for Mistral 429 (rate limit) errors.
# Free-tier Mistral keys allow very few requests/second, and this
# pipeline fires 4 LLM calls back-to-back, so we retry with an
# increasing delay instead of failing the whole stage immediately.
MAX_RETRIES = 5
BASE_RETRY_DELAY = 4          # seconds, doubles each retry (4, 8, 16, 32, 64)
INTER_STAGE_DELAY = 3         # seconds to wait between pipeline stages
MIN_TOPIC_LENGTH = 3

QUICK_TOPICS = [
    "Impact of AI on the job market",
    "The future of renewable energy",
    "Remote work productivity trends",
    "Quantum computing breakthroughs",
    "Global semiconductor supply chain",
]


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / CUSTOM CSS
# ============================================================
DARK_THEME_VARS = {
    "bg": "#0f1117",
    "bg_gradient_end": "#14161f",
    "card_bg": "#1b1e29",
    "card_border": "#2a2e3d",
    "text_muted": "#9aa0ac",
}

LIGHT_THEME_VARS = {
    "bg": "#f7f8fa",
    "bg_gradient_end": "#eef0f4",
    "card_bg": "#ffffff",
    "card_border": "#e1e4ea",
    "text_muted": "#5b6270",
}


def inject_custom_css(theme: str = "dark") -> None:
    """Injects theme-aware CSS into the Streamlit app."""
    v = DARK_THEME_VARS if theme == "dark" else LIGHT_THEME_VARS

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {v['bg']} 0%, {v['bg_gradient_end']} 100%);
        }}
        .hero-title {{
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7ee8fa 0%, #eec0c6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0rem;
        }}
        .hero-subtitle {{
            color: {v['text_muted']};
            font-size: 1.05rem;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }}
        .agent-card {{
            background-color: {v['card_bg']};
            border: 1px solid {v['card_border']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }}
        .agent-card h4 {{
            margin: 0 0 0.3rem 0;
            font-size: 1rem;
        }}
        .badge-pending {{
            color: {v['text_muted']};
            background-color: {v['card_border']};
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
        }}
        .badge-running {{
            color: #1a1f2b;
            background-color: #f5d76e;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
        }}
        .badge-done {{
            color: #0f2b1d;
            background-color: #6ee7a8;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
        }}
        .badge-error {{
            color: #2b0f0f;
            background-color: #f28b82;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
        }}
        .metric-box {{
            background-color: {v['card_bg']};
            border: 1px solid {v['card_border']};
            border-radius: 12px;
            padding: 0.8rem 1rem;
            text-align: center;
        }}
        .metric-box h3 {{
            margin: 0.1rem 0;
        }}
        .history-item {{
            background-color: {v['card_bg']};
            border: 1px solid {v['card_border']};
            border-radius: 8px;
            padding: 0.5rem 0.7rem;
            margin-bottom: 0.4rem;
            font-size: 0.85rem;
        }}
        .chip {{
            display: inline-block;
            background-color: {v['card_bg']};
            border: 1px solid {v['card_border']};
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.8rem;
            margin: 2px;
        }}
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state() -> None:
    """Initializes all session-state keys used across the app."""
    defaults: Dict[str, Any] = {
        "history": [],           # list[dict]: topic, timestamp, state, elapsed
        "current_state": None,   # dict with search_results, scraped_content, report, feedback
        "is_running": False,
        "last_topic": "",
        "elapsed_time": 0.0,
        "theme": "dark",
        "total_runs": 0,
        "total_errors": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# RETRY / RATE-LIMIT HANDLING
# ============================================================
def is_rate_limit_error(exc: Exception) -> bool:
    """Detects a Mistral 429 / rate-limited error from the exception text."""
    msg = str(exc).lower()
    return "429" in msg or "rate_limited" in msg or "rate limit" in msg


def call_with_retry(func, *args, status_slot=None, **kwargs):
    """
    Calls func(*args, **kwargs), automatically retrying with exponential
    backoff if the failure looks like a Mistral rate-limit (429) error.
    Re-raises immediately for any other kind of exception.
    """
    delay = BASE_RETRY_DELAY
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not is_rate_limit_error(e) or attempt == MAX_RETRIES:
                raise
            if status_slot is not None:
                status_slot.markdown(
                    f'<span class="badge-running">Rate limited — retry {attempt}/{MAX_RETRIES} '
                    f'in {delay}s…</span>',
                    unsafe_allow_html=True,
                )
            time.sleep(delay)
            delay *= 2  # exponential backoff

    # Should never reach here, but keeps type-checkers happy.
    raise last_exc


# ============================================================
# PIPELINE STEPS (mirrors pipeline.py logic, with live UI hooks)
# ============================================================
def run_step_search(topic: str) -> str:
    """Invokes the search agent and returns its final message content."""
    search_agent = build_search_agent()
    result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    return result["messages"][-1].content


def run_step_read(topic: str, search_results: str) -> str:
    """Invokes the reader agent to scrape the most relevant URL from search_results."""
    reader_agent = build_reader_agent()
    result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{search_results}"
        )]
    })
    return result["messages"][-1].content


def run_step_write(topic: str, search_results: str, scraped_content: str) -> str:
    """Invokes the writer chain to draft a structured report."""
    research_combined = (
        f"SEARCH RESULTS:\n{search_results}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{scraped_content}"
    )
    return writer_chain.invoke({"topic": topic, "research": research_combined})


def run_step_critique(report: str) -> str:
    """Invokes the critic chain to review and score the report."""
    return critic_chain.invoke({"report": report})


def set_badge(slot, kind: str) -> None:
    """Updates a status placeholder with the given badge style."""
    labels = {
        "pending": ("badge-pending", "Pending"),
        "running": ("badge-running", "Running…"),
        "done": ("badge-done", "Done"),
        "error": ("badge-error", "Error"),
    }
    css_class, label = labels[kind]
    slot.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


def run_full_pipeline(topic: str, status_slots: Dict[str, Any], progress_bar) -> Dict[str, Any]:
    """
    Runs all 4 pipeline stages sequentially, updating status badges and
    the progress bar live as each stage starts/completes. Returns the
    accumulated state dict, never raising — failures are captured inline
    so the rest of the pipeline can still proceed with partial data.
    """
    state: Dict[str, Any] = {
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "errors": [],
    }

    # --- Step 1: Search ---
    set_badge(status_slots["search"], "running")
    try:
        state["search_results"] = call_with_retry(
            run_step_search, topic, status_slot=status_slots["search"]
        )
        set_badge(status_slots["search"], "done")
    except Exception as e:
        state["search_results"] = f"Search failed: {e}"
        state["errors"].append(f"Search agent: {e}")
        set_badge(status_slots["search"], "error")
    progress_bar.progress(25)
    time.sleep(INTER_STAGE_DELAY)

    # --- Step 2: Read / Scrape ---
    set_badge(status_slots["read"], "running")
    try:
        state["scraped_content"] = call_with_retry(
            run_step_read, topic, state["search_results"], status_slot=status_slots["read"]
        )
        set_badge(status_slots["read"], "done")
    except Exception as e:
        state["scraped_content"] = f"Scraping failed: {e}"
        state["errors"].append(f"Reader agent: {e}")
        set_badge(status_slots["read"], "error")
    progress_bar.progress(55)
    time.sleep(INTER_STAGE_DELAY)

    # --- Step 3: Write ---
    set_badge(status_slots["write"], "running")
    try:
        state["report"] = call_with_retry(
            run_step_write,
            topic,
            state["search_results"],
            state["scraped_content"],
            status_slot=status_slots["write"],
        )
        set_badge(status_slots["write"], "done")
    except Exception as e:
        state["report"] = f"Report generation failed: {e}"
        state["errors"].append(f"Writer chain: {e}")
        set_badge(status_slots["write"], "error")
    progress_bar.progress(80)
    time.sleep(INTER_STAGE_DELAY)

    # --- Step 4: Critique ---
    set_badge(status_slots["critique"], "running")
    try:
        state["feedback"] = call_with_retry(
            run_step_critique, state["report"], status_slot=status_slots["critique"]
        )
        set_badge(status_slots["critique"], "done")
    except Exception as e:
        state["feedback"] = f"Critique failed: {e}"
        state["errors"].append(f"Critic chain: {e}")
        set_badge(status_slots["critique"], "error")
    progress_bar.progress(100)

    return state


# ============================================================
# HELPERS
# ============================================================
def word_count(text: Optional[str]) -> int:
    """Returns the number of whitespace-separated words in text."""
    if not text:
        return 0
    return len(text.split())


def extract_score(feedback: Optional[str]) -> str:
    """Pulls a 'Score: X/10' line out of the critic feedback if present."""
    if not feedback:
        return "N/A"
    for line in feedback.splitlines():
        if line.strip().lower().startswith("score"):
            return line.strip()
    return "N/A"


def state_to_json(topic: str, state: Dict[str, Any]) -> str:
    """Serializes a run's topic + state to a pretty JSON string for export."""
    payload = {
        "topic": topic,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "search_results": state.get("search_results", ""),
        "scraped_content": state.get("scraped_content", ""),
        "report": state.get("report", ""),
        "feedback": state.get("feedback", ""),
        "errors": state.get("errors", []),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def validate_topic(topic: str) -> Optional[str]:
    """Returns an error message if the topic is invalid, else None."""
    if not topic or not topic.strip():
        return "Please enter a topic before running the pipeline."
    if len(topic.strip()) < MIN_TOPIC_LENGTH:
        return f"Topic is too short — please use at least {MIN_TOPIC_LENGTH} characters."
    return None


def render_quick_topics() -> Optional[str]:
    """Renders quick-pick topic chips as buttons. Returns a topic if clicked."""
    st.caption("Or try a quick topic:")
    cols = st.columns(len(QUICK_TOPICS))
    chosen = None
    for col, topic in zip(cols, QUICK_TOPICS):
        with col:
            if st.button(topic, key=f"quick_{topic}", use_container_width=True):
                chosen = topic
    return chosen


def render_agent_pipeline_cards() -> Dict[str, Any]:
    """Renders the 4 agent-stage cards and returns placeholders for live status badges."""
    cols = st.columns(4)
    labels = [
        ("search", "🔍 Search Agent", "Scans the web for recent, reliable sources."),
        ("read", "📖 Reader Agent", "Scrapes the most relevant page for depth."),
        ("write", "✍️ Writer Chain", "Drafts a structured research report."),
        ("critique", "🧐 Critic Chain", "Scores and critiques the final report."),
    ]
    slots: Dict[str, Any] = {}
    for col, (key, title, desc) in zip(cols, labels):
        with col:
            st.markdown(
                f"""
                <div class="agent-card">
                    <h4>{title}</h4>
                    <p style="color:#9aa0ac; font-size:0.8rem; margin-bottom:0.5rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            slots[key] = st.empty()
            set_badge(slots[key], "pending")
    return slots


def add_to_history(topic: str, state: Dict[str, Any], elapsed: float) -> None:
    """Prepends a completed run to the session history, capped at MAX_HISTORY_ITEMS."""
    st.session_state.history.insert(0, {
        "topic": topic,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "state": state,
        "elapsed": elapsed,
    })
    st.session_state.history = st.session_state.history[:MAX_HISTORY_ITEMS]
    st.session_state.total_runs += 1
    st.session_state.total_errors += len(state.get("errors", []))


def render_history_item(item: Dict[str, Any], index: int) -> None:
    """Renders a single history row with a reload button."""
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown(
            f"""
            <div class="history-item">
            <b>{item['topic'][:40]}</b><br>
            <span style="color:#9aa0ac;">{item['timestamp']} · {item['elapsed']:.1f}s</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button("↺", key=f"reload_{index}", help="Reload this result"):
            st.session_state.current_state = item["state"]
            st.session_state.last_topic = item["topic"]
            st.rerun()


def render_sidebar() -> None:
    """Renders the full sidebar: about, theme toggle, analytics, and history."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        theme_choice = st.radio(
            "Theme",
            options=["dark", "light"],
            index=0 if st.session_state.theme == "dark" else 1,
            horizontal=True,
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown("---")
        st.markdown("### 📈 Session Analytics")
        a1, a2 = st.columns(2)
        with a1:
            st.metric("Total Runs", st.session_state.total_runs)
        with a2:
            st.metric("Total Errors", st.session_state.total_errors)

        st.markdown("---")
        st.markdown("### 🧠 About this app")
        st.write(
            "This app orchestrates a 4-stage multi-agent pipeline built with "
            "**LangChain** and **Mistral**:\n\n"
            "1. **Search Agent** — finds relevant sources on the web\n"
            "2. **Reader Agent** — scrapes the most relevant URL for detail\n"
            "3. **Writer Chain** — drafts a structured report\n"
            "4. **Critic Chain** — reviews and scores the report\n"
        )

        with st.expander("❓ How does this work?"):
            st.write(
                "Each stage is a separate LangChain component:\n\n"
                "- The **search** and **reader** stages are tool-calling agents "
                "that can decide when to call `web_search` / `scrape_url`.\n"
                "- The **writer** and **critic** stages are plain prompt → LLM → "
                "string chains with no tools attached.\n\n"
                "All four stages share one Mistral chat model instance defined in "
                "`agents.py`."
            )

        st.markdown("---")
        st.markdown("### 🕘 Run History")

        if not st.session_state.history:
            st.caption("No runs yet. Your past research topics will show up here.")
        else:
            for i, item in enumerate(st.session_state.history):
                render_history_item(item, i)

            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.total_runs = 0
                st.session_state.total_errors = 0
                st.rerun()

        st.markdown("---")
        st.caption("Powered by LangChain + Mistral · Built with Streamlit")


def render_metrics_row(state: Dict[str, Any]) -> None:
    """Renders the 4-column metrics dashboard above the results tabs."""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-box"><h3>{word_count(state.get("report"))}</h3>'
            f'<span style="color:#9aa0ac;">Report words</span></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><h3>{word_count(state.get("scraped_content"))}</h3>'
            f'<span style="color:#9aa0ac;">Scraped words</span></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-box"><h3>{extract_score(state.get("feedback"))}</h3>'
            f'<span style="color:#9aa0ac;">Critic score</span></div>',
            unsafe_allow_html=True,
        )
    with m4:
        err_count = len(state.get("errors", []))
        badge = "✅ 0" if err_count == 0 else f"⚠️ {err_count}"
        st.markdown(
            f'<div class="metric-box"><h3>{badge}</h3>'
            f'<span style="color:#9aa0ac;">Errors</span></div>',
            unsafe_allow_html=True,
        )


def render_export_buttons(topic: str, state: Dict[str, Any]) -> None:
    """Renders download buttons for Markdown report and full JSON export."""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Report (.md)",
            data=state.get("report", ""),
            file_name=f"{topic.replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download Full Run (.json)",
            data=state_to_json(topic, state),
            file_name=f"{topic.replace(' ', '_')}_run.json",
            mime="application/json",
            use_container_width=True,
        )


def render_results(topic: str, state: Dict[str, Any]) -> None:
    """Renders the full results section: metrics, error expander, and tabs."""
    st.markdown("## 📊 Results")

    render_metrics_row(state)

    if state.get("errors"):
        with st.expander("⚠️ View errors encountered during this run"):
            for err in state["errors"]:
                st.error(err)

    st.markdown("")

    tab_report, tab_critic, tab_search, tab_scrape = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        st.markdown(state.get("report", "_No report generated._"))
        render_export_buttons(topic, state)

    with tab_critic:
        st.markdown(state.get("feedback", "_No feedback generated._"))

    with tab_search:
        st.text_area(
            "Raw search results",
            value=state.get("search_results", "No search results."),
            height=350,
        )

    with tab_scrape:
        st.text_area(
            "Scraped page content",
            value=state.get("scraped_content", "No scraped content."),
            height=350,
        )


def render_compare_panel(history: List[Dict[str, Any]]) -> None:
    """Renders an optional comparison table across the last few runs."""
    if len(history) < 2:
        return

    with st.expander("📐 Compare recent runs"):
        rows = []
        for item in history[:5]:
            state = item["state"]
            rows.append({
                "Topic": item["topic"][:40],
                "Time (s)": round(item["elapsed"], 1),
                "Report words": word_count(state.get("report")),
                "Critic score": extract_score(state.get("feedback")),
                "Errors": len(state.get("errors", [])),
            })
        st.table(rows)


# ============================================================
# MAIN APP
# ============================================================
def render_header() -> None:
    """Renders the hero title and subtitle."""
    st.markdown(f'<div class="hero-title">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Search Agent → Reader Agent → Writer → Critic, '
        'working together to turn a topic into a polished, reviewed report.</div>',
        unsafe_allow_html=True,
    )


def render_input_form() -> Any:
    """Renders the topic input form. Returns the submit button state and topic."""
    with st.form("research_form", clear_on_submit=False):
        col1, col2 = st.columns([5, 1])
        with col1:
            topic = st.text_input(
                "Research topic",
                value=st.session_state.last_topic,
                placeholder="e.g. Impact of AI on the job market",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("🚀 Run Pipeline", use_container_width=True)
    return submitted, topic


def execute_pipeline_run(topic: str, status_slots: Dict[str, Any], progress_bar) -> None:
    """Runs the pipeline for a validated topic, updating session state and history."""
    st.session_state.is_running = True
    st.session_state.last_topic = topic

    start_time = time.time()
    with st.spinner("Agents are working on your topic..."):
        state = run_full_pipeline(topic, status_slots, progress_bar)
    elapsed = time.time() - start_time

    st.session_state.current_state = state
    st.session_state.elapsed_time = elapsed
    st.session_state.is_running = False

    add_to_history(topic, state, elapsed)

    if state.get("errors"):
        st.warning(f"Pipeline finished in {elapsed:.1f}s with {len(state['errors'])} issue(s).")
    else:
        st.success(f"Pipeline finished successfully in {elapsed:.1f}s ✅")


def main() -> None:
    """Application entry point."""
    init_session_state()
    inject_custom_css(theme=st.session_state.theme)

    render_header()

    submitted, topic_input = render_input_form()
    quick_pick = render_quick_topics()

    # A quick-pick chip triggers a run using that topic even without the form.
    effective_topic = topic_input
    trigger_run = submitted
    if quick_pick:
        effective_topic = quick_pick
        trigger_run = True

    st.markdown("")

    status_slots = render_agent_pipeline_cards()
    progress_bar = st.progress(0)

    if trigger_run:
        error_msg = validate_topic(effective_topic)
        if error_msg:
            st.warning(error_msg)
        else:
            execute_pipeline_run(effective_topic.strip(), status_slots, progress_bar)

    st.markdown("---")

    if st.session_state.current_state:
        render_results(
            st.session_state.last_topic or effective_topic or "Untitled Topic",
            st.session_state.current_state,
        )
        render_compare_panel(st.session_state.history)
    else:
        st.info("Enter a topic above, pick a quick suggestion, or click **Run Pipeline** to get started.")

    render_sidebar()


if __name__ == "__main__":
    main()