"""
Website Insights Agent — main agent loop.

Uses raw Gemini API calls (no ADK/LangGraph) so it's straightforward
to port tool definitions to AWS Bedrock later.
"""

import os
import yaml
import google.generativeai as genai

from agent.tools.s3_reader import load_ga4_data, load_search_console_data
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


def run_agent(config_path: str = "config.yaml") -> str:
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
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=cfg["model"]["name"],
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": cfg["model"]["temperature"],
            "max_output_tokens": cfg["model"]["max_output_tokens"],
        },
    )

    # --- Tool calls: load data ---
    ga4_df = load_ga4_data(
        bucket=cfg["s3"]["bucket"],
        prefix=cfg["s3"]["paths"]["ga4"],
    )
    sc_df = load_search_console_data(
        bucket=cfg["s3"]["bucket"],
        prefix=cfg["s3"]["paths"]["search_console"],
    )

    # --- Build prompt context ---
    context = build_analysis_context(
        ga4_df=ga4_df,
        sc_df=sc_df,
        site_url=cfg["site"]["url"],
    )

    user_prompt = (
        f"Here is the latest website data:\n\n{context}\n\n"
        "Please provide your top recommendations."
    )

    # --- Gemini call ---
    response = model.generate_content(user_prompt)
    recommendations = response.text

    # --- Tool call: send email ---
    body = format_email_body(cfg["site"]["url"], recommendations)
    send_recommendations(
        lambda_url=cfg["email"]["lambda_url"],
        recipient=cfg["email"]["recipient"],
        sender=cfg["email"]["sender"],
        subject=f"Website Insights: {cfg['site']['url']}",
        body=body,
    )

    return recommendations


if __name__ == "__main__":
    result = run_agent()
    print(result)
