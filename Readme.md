# 🔎 Multi-Agent Research System

A multi-agent research assistant built with **LangChain** and **Mistral AI**. Give it a topic, and four agents/chains work together to search the web, read the most relevant source in depth, write a structured report, and critique it — all visible through a live Streamlit UI.

**Live demo:** [multi-agent-research-system-vzra.onrender.com](https://multi-agent-research-system-vzra.onrender.com/)
 -->

---

## Screenshots

<!-- Drop your actual app screenshot in an /assets folder and reference it, e.g.: -->
<!-- ![App running](assets/app-screenshot.png) -->

![Pipeline overview](assets/pipeline-diagram.png)

*The four-stage pipeline — Search, Reader, Writer, Critic — all sharing a single Mistral model instance.*

---

## How it works

```
Topic
  │
  ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  🔍 Search Agent  │ ──▶ │  📖 Reader Agent  │ ──▶ │  ✍️ Writer Chain  │ ──▶ │  🧐 Critic Chain  │
│  finds sources    │     │  scrapes the      │     │  drafts a         │     │  scores and       │
│  on the web       │     │  most relevant    │     │  structured       │     │  critiques the    │
│                    │     │  URL for depth    │     │  report           │     │  report           │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

| Stage | Type | Tool used | Purpose |
|---|---|---|---|
| **Search Agent** | Tool-calling agent (`create_agent`) | `web_search` (Tavily) | Finds recent, relevant sources for the topic |
| **Reader Agent** | Tool-calling agent (`create_agent`) | `scrape_url` (requests + BeautifulSoup) | Picks the best URL from search results and scrapes it |
| **Writer Chain** | LCEL chain (`prompt \| llm \| StrOutputParser`) | — | Drafts a structured report (Introduction, Key Findings, Conclusion, Sources) |
| **Critic Chain** | LCEL chain (`prompt \| llm \| StrOutputParser`) | — | Scores the report out of 10 with strengths, areas to improve, and a verdict |

All four stages share a single `ChatMistralAI` model instance (`mistral-large-latest`).

---

## Project structure

```
multi-agent-project/
├── .env                # API keys (not committed to version control)
├── requirements.txt    # Python dependencies
├── tools.py             # @tool-decorated functions: web_search, scrape_url
├── agents.py            # Builds the two agents + two chains, shared LLM instance
├── pipeline.py           # Terminal-based orchestrator (runs all 4 stages, prints to console)
├── app.py                # Streamlit UI — live status, metrics, tabs, history, exports
└── README.md
```

---

## Setup

### 1. Clone / open the project folder
```bash
cd multi-agent-project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install directly:
```bash
pip install langchain langgraph langchain-mistralai langchain-core \
            tavily-python requests beautifulsoup4 python-dotenv rich streamlit
```

> ⚠️ **Version compatibility matters.** `langchain.agents.create_agent` depends on matching versions of `langchain` and `langgraph`. If you see `ImportError: cannot import name 'InjectedState' from 'langgraph.prebuilt'`, upgrade both together:
> ```bash
> pip install -U langchain langgraph langchain-core
> ```

### 4. Add your API keys
Create a `.env` file in the project root:
```env
TAVILY_API_KEY=your_tavily_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

- Get a Tavily key at [tavily.com](https://tavily.com)
- Get a Mistral key at [console.mistral.ai](https://console.mistral.ai)

---

## Running the project

### Option A — Terminal pipeline
```bash
python pipeline.py
```
Prompts you for a topic, then prints each stage's output directly to the console.

### Option B — Streamlit UI (recommended)
```bash
streamlit run app.py
```
Then open the URL it prints (usually `http://localhost:8501`).

Or just use the **[live version](https://multi-agent-research-system-vzra.onrender.com/)** — no setup needed.

The UI gives you:
- Live per-agent status badges (Pending → Running → Done/Error)
- A progress bar and metrics dashboard (word counts, critic score, error count)
- Tabbed results: Final Report / Critic Feedback / Search Results / Scraped Content
- Run history with reload, and a "compare recent runs" table
- Quick topic chips, light/dark theme toggle, and Markdown/JSON export

---

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| Mistral model | `agents.py` → `llm = ChatMistralAI(model=...)` | `mistral-large-latest` | Swap to `mistral-small-latest` for a cheaper/faster model |
| Temperature | `agents.py` | `0` | Higher = more creative, less deterministic |
| Retry attempts | `app.py` → `MAX_RETRIES` | `5` | How many times a rate-limited call is retried |
| Retry backoff | `app.py` → `BASE_RETRY_DELAY` | `4` seconds | Doubles each retry (4s → 8s → 16s → 32s → 64s) |
| Delay between stages | `app.py` → `INTER_STAGE_DELAY` | `3` seconds | Extra buffer between the 4 pipeline stages |

---

## Troubleshooting

### `429 rate_limited` errors on Reader/Writer/Critic stages
Free-tier Mistral API keys allow very few requests per second. `app.py` already retries with exponential backoff, but if you're still hitting the limit after all retries:
- Check your usage/quota at [console.mistral.ai](https://console.mistral.ai)
- Increase `BASE_RETRY_DELAY` and `INTER_STAGE_DELAY` in `app.py`
- Consider upgrading your Mistral plan for higher rate limits

### `ImportError: cannot import name 'create_tool_calling_agent' from 'langchain.agents'`
This means `tools.py` still has a leftover import from an older LangChain agent-building pattern. `tools.py` only needs:
```python
from langchain.tools import tool
```
It should **not** import `create_tool_calling_agent`, `AgentExecutor`, `ChatMistralAI`, or `ChatPromptTemplate` — those belong in `agents.py`.

### `ImportError: cannot import name 'InjectedState' from 'langgraph.prebuilt'`
Version mismatch between `langchain` and `langgraph`. Fix with:
```bash
pip install -U langchain langgraph langchain-core
```

### `MISTRAL_API_KEY is not set` / agent fails silently
Confirm your `.env` is in the same folder you're running commands from, and check it loads:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('MISTRAL_API_KEY'))"
```

### Reader agent never calls `scrape_url`
Occasionally a smaller/faster model decides it "already knows enough" and skips the tool call. Stick with `mistral-large-latest` for the agent stages, or tighten the reader agent's prompt to require tool use.

---

## Roadmap ideas

- [ ] Swap Tavily for a different search provider
- [ ] Add PDF export of the final report
- [ ] Support multiple source scraping instead of just the top URL
- [ ] Add a settings panel to change model/temperature from the UI
- [ ] Cache repeated topic searches to save API calls

---

## Tech stack

- [LangChain](https://python.langchain.com/) — agent + chain orchestration
- [Mistral AI](https://mistral.ai/) — LLM (`mistral-large-latest`)
- [Tavily](https://tavily.com/) — web search API
- [Streamlit](https://streamlit.io/) — UI
- BeautifulSoup + Requests — web scraping
