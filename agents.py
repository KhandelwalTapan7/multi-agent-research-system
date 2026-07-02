"""
agents.py
============================================================
Defines the four building blocks of the multi-agent research
pipeline used by pipeline.py and app.py:

    1. build_search_agent()  -> tool-calling agent (web_search)
    2. build_reader_agent()  -> tool-calling agent (scrape_url)
    3. writer_chain          -> prompt | llm | StrOutputParser
    4. critic_chain          -> prompt | llm | StrOutputParser

All four share a single ChatMistralAI instance.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

load_dotenv()


# ============================================================
# Model setup
# ============================================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. Add it to your .env file, e.g.:\n"
        "MISTRAL_API_KEY=your_key_here"
    )

llm = ChatMistralAI(
    model="mistral-large-latest",   # or "mistral-small-latest" for cheaper/faster
    api_key=MISTRAL_API_KEY,
    temperature=0,
)


# ============================================================
# Agent 1 — Search Agent
# ============================================================
def build_search_agent():
    """Tool-calling agent that can use web_search to gather information."""
    return create_agent(
        model=llm,
        tools=[web_search],
    )


# ============================================================
# Agent 2 — Reader Agent
# ============================================================
def build_reader_agent():
    """Tool-calling agent that can use scrape_url to read a page in depth."""
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )


# ============================================================
# Writer Chain
# ============================================================
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ============================================================
# Critic Chain
# ============================================================
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()