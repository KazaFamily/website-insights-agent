"""
Website Insights Agent — agentic loop.

The agent receives a prompt, has access to tools, and decides what to call.
Loop runs until the model stops requesting tools (no function_call in response).

To port to Bedrock: swap get_client() and get_gemini_tool_config() for their
Bedrock equivalents. The tool implementations and memory are untouched.
"""

import os
import yaml
import pandas as pd
from google.genai import types

from agent.llm.client import get_client
from agent.llm.tool_registry import get_tool_definitions, get_gemini_tool_config
from agent.tools.page_fetcher import fetch_page_metadata, fetch_page_text
from agent.tools.memory import get_memory, save_memory
from agent.tools.s3_reader import (
    load_ga4_pages_data,
    load_ga4_traffic_data,
    load_search_console_pages_data,
    load_search_console_queries_data,
)
from agent.tools.analyzer import build_analysis_context
from agent.tools.email_sender import send_recommendations, format_email_body

_LOCAL = os.environ.get("LOCAL", "true").lower() == "true"

SYSTEM_PROMPT = """\
You are a website performance analyst agent for {site_url}.

You have access to tools. Use them in this general order:
1. Call get_memory to see what has already been analyzed and recommended.
2. Call load_analytics_data to get the current GA4 and Search Console summary.
3. Identify 2-3 pages that deserve deeper investigation based on the data.
4. For each page, call fetch_page_metadata to audit its SEO metadata.
5. Optionally call fetch_page_text if content quality is in question.
6. Call save_memory with your final recommendations and the pages you analyzed.
7. Return a final prioritized recommendation report.

Be specific. Name pages and queries. Write for a non-technical site owner.
Keep the final report under 600 words.\
"""


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_analytics_data_tool(cfg: dict, local: bool = _LOCAL) -> str:
    """Load sample CSVs (local=True) or S3 data and return the analysis context string."""
    if local:
        ga4_pages_df = pd.read_csv(
            "data/sample/Pages_and_screens_Page_title_and_screen_class_90_days.csv",
            skiprows=9,
        )
        ga4_traffic_df = pd.read_csv(
            "data/sample/Traffic_acquisition_Session_source_medium_90_days.csv",
            skiprows=9,
        )
        sc_pages_df = pd.read_csv("data/sample/search_console_pages_3_months.csv")
        sc_queries_df = pd.read_csv("data/sample/search_console_queries_3_months.csv")
    else:
        ga4_pages_df = load_ga4_pages_data(
            bucket=cfg["s3"]["bucket"], prefix=cfg["s3"]["paths"]["ga4_pages"]
        )
        ga4_traffic_df = load_ga4_traffic_data(
            bucket=cfg["s3"]["bucket"], prefix=cfg["s3"]["paths"]["ga4_traffic"]
        )
        sc_pages_df = load_search_console_pages_data(
            bucket=cfg["s3"]["bucket"], prefix=cfg["s3"]["paths"]["search_console_pages"]
        )
        sc_queries_df = load_search_console_queries_data(
            bucket=cfg["s3"]["bucket"], prefix=cfg["s3"]["paths"]["search_console_queries"]
        )

    return build_analysis_context(
        ga4_pages_df=ga4_pages_df,
        ga4_traffic_df=ga4_traffic_df,
        sc_pages_df=sc_pages_df,
        sc_queries_df=sc_queries_df,
        site_url=cfg["site"]["url"],
    )


def run_agent(config_path: str = "config.yaml", local: bool = _LOCAL) -> str:
    """
    Agentic loop: model calls tools until it produces a final text response.
    Capped at 10 iterations to prevent runaway calls.
    """
    cfg = load_config(config_path)
    client = get_client("gemini")

    # Seed the agent with memory context
    memory = get_memory()
    memory_summary = (
        f"Last run: {memory['last_run'].get('timestamp', 'never')}. "
        f"Pages previously analyzed: {list(memory['page_analyses'].keys()) or 'none'}."
    )

    system_prompt = SYSTEM_PROMPT.format(site_url=cfg["site"]["url"])
    system_prompt += f"\n\nMemory context: {memory_summary}"

    tool_definitions = get_tool_definitions()
    tool_config = get_gemini_tool_config(tool_definitions)

    # Capture cfg and local in the dispatch closure
    def _load_analytics(**_):
        return load_analytics_data_tool(cfg, local)

    TOOL_DISPATCH = {
        "fetch_page_metadata": fetch_page_metadata,
        "fetch_page_text": fetch_page_text,
        "get_memory": lambda **_: get_memory(),
        "save_memory": save_memory,
        "load_analytics_data": _load_analytics,
    }

    messages = [{"role": "user", "parts": [{"text": "Please analyze the website and provide recommendations."}]}]

    for _ in range(10):
        response = client.models.generate_content(
            model=cfg["model"]["name"],
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=cfg["model"]["temperature"],
                max_output_tokens=cfg["model"]["max_output_tokens"],
                tools=[tool_config],
            ),
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check for a function call in any part
        function_call = next(
            (p.function_call for p in parts if hasattr(p, "function_call") and p.function_call),
            None,
        )

        if function_call is None:
            # No more tool calls — extract final text
            final_text = "".join(
                p.text for p in parts if hasattr(p, "text") and p.text
            )
            return final_text

        # Append the model's response to messages
        messages.append({"role": "model", "parts": [{"function_call": function_call}]})

        # Dispatch the tool call
        tool_name = function_call.name
        tool_args = dict(function_call.args) if function_call.args else {}
        tool_fn = TOOL_DISPATCH.get(tool_name)
        if tool_fn is None:
            tool_result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                tool_result = tool_fn(**tool_args)
            except Exception as e:
                tool_result = {"error": str(e)}

        messages.append({
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": tool_name,
                    "response": {"result": tool_result},
                }
            }],
        })

    return "Agent reached maximum iterations without producing a final response."


if __name__ == "__main__":
    result = run_agent(local=_LOCAL)
    print(result)
