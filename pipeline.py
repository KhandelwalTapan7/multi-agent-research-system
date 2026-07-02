from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # ---------------------------------------------------------
    # Step 1 - Search agent
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    try:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result['messages'][-1].content
    except Exception as e:
        print(f"[ERROR] Search agent failed: {e}")
        state["search_results"] = f"Search failed: {e}"

    print("\n search result:\n", state["search_results"])

    # ---------------------------------------------------------
    # Step 2 - Reader agent
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    try:
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results']}"
            )]
        })

        # Debug: show full message trace so you can confirm scrape_url was actually called
        print("\n[DEBUG] Reader agent message trace:")
        for m in reader_result['messages']:
            print(f"  {type(m).__name__}: {getattr(m, 'content', None)}")

        state["scraped_content"] = reader_result['messages'][-1].content
    except Exception as e:
        print(f"[ERROR] Reader agent failed: {e}")
        state["scraped_content"] = f"Scraping failed: {e}"

    print("\nscraped content:\n", state["scraped_content"])

    # ---------------------------------------------------------
    # Step 3 - Writer chain
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    try:
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })
    except Exception as e:
        print(f"[ERROR] Writer chain failed: {e}")
        state["report"] = f"Report generation failed: {e}"

    print("\n Final Report\n", state["report"])

    # ---------------------------------------------------------
    # Step 4 - Critic chain
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("step 4 - critic is reviewing the report")
    print("=" * 50)

    try:
        state["feedback"] = critic_chain.invoke({
            "report": state["report"]
        })
    except Exception as e:
        print(f"[ERROR] Critic chain failed: {e}")
        state["feedback"] = f"Critique failed: {e}"

    print("\n critic report\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)