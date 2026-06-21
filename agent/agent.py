"""
Website Insights Agent — main agent loop.

Uses raw Gemini API calls (no ADK/LangGraph) so it's straightforward
to port tool definitions to AWS Bedrock later.
"""

from http import client
import os
import yaml
import pandas as pd
from google import genai
from google.genai import types

from agent.tools.s3_reader import load_ga4_pages_data, load_ga4_traffic_data, load_search_console_pages_data, load_search_console_queries_data
from agent.tools.analyzer import build_analysis_context
from agent.tools.email_sender import send_recommendations, format_email_body


SYSTEM_PROMPT = """You are a website performance analyst. You will be given
traffic and search data for a website. Your job is to:
1. Identify the top 3-5 actionable insights from the data.
2. Prioritize recommendations by expected impact.
3. Keep recommendations specific and concrete — cite page paths or queries
   where relevant.
4. Write in plain language suitable for a non-technical site owner.
"""


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_agent(config_path: str = "config.yaml", local=False) -> str:
    """
    Full agent loop:
    1. Load data from S3
    2. Summarize data into prompt context
    3. Call Gemini for recommendations
    4. Send recommendations via email Lambda
    5. Return the recommendations string
    """
    cfg = load_config(config_path)

    # Configure Gemini — API key from environment variable
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # --- Tool calls: load data ---
    if local:
        ga4_pages_df = pd.read_csv("data/sample/Pages_and_screens_Page_title_and_screen_class_90_days.csv", skiprows=9)
        ga4_traffic_df = pd.read_csv("data/sample/Traffic_acquisition_Session_source_medium_90_days.csv", skiprows=9)
        sc_pages_df = pd.read_csv("data/sample/search_console_pages_3_months.csv")
        sc_queries_df = pd.read_csv("data/sample/search_console_queries_3_months.csv")
    else:
        ga4_pages_df = load_ga4_pages_data(
            bucket=cfg["s3"]["bucket"],
            prefix=cfg["s3"]["paths"]["ga4_pages"],
        )
        ga4_traffic_df = load_ga4_traffic_data(
            bucket=cfg["s3"]["bucket"],
            prefix=cfg["s3"]["paths"]["ga4_traffic"],
        )
        sc_pages_df = load_search_console_pages_data(
            bucket=cfg["s3"]["bucket"],
            prefix=cfg["s3"]["paths"]["search_console_pages"],
        )
        sc_queries_df = load_search_console_queries_data(
            bucket=cfg["s3"]["bucket"],
            prefix=cfg["s3"]["paths"]["search_console_queries"],
        )

    # --- Build prompt context ---
    context = build_analysis_context(
        ga4_pages_df=ga4_pages_df,
        ga4_traffic_df=ga4_traffic_df,
        sc_pages_df=sc_pages_df,
        sc_queries_df=sc_queries_df,
        site_url=cfg["site"]["url"],
    )
    user_prompt = (
        f"Here is the latest website data:\n\n{context}\n\n"
        "Please provide your top recommendations."
    )

    # --- Gemini call ---
    response = client.models.generate_content(
        model=cfg["model"]["name"],
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=cfg["model"]["temperature"],
            max_output_tokens=cfg["model"]["max_output_tokens"],
        ),
    )
    recommendations = response.text

    # --- Tool call: send email ---
    #body = format_email_body(cfg["site"]["url"], recommendations)
    #send_recommendations(
    #    lambda_url=cfg["email"]["lambda_url"],
    #    recipient=cfg["email"]["recipient"],
    #    sender=cfg["email"]["sender"],
    #    subject=f"Website Insights: {cfg['site']['url']}",
    #    body=body,
    #)

    return recommendations


if __name__ == "__main__":
    result = run_agent(local=True)
    print(result)
