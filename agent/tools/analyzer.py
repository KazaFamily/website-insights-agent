"""
Summarizes GA4 and Search Console DataFrames into plain-text context
that the agent can pass to Gemini.
"""

import pandas as pd


def summarize_ga4(df: pd.DataFrame) -> str:
    """
    Return a text summary of GA4 data suitable for inclusion in a prompt.
    Expected columns: date, page_path, sessions, bounce_rate, avg_session_duration
    Adjust column names to match your actual GA4 export schema.
    """
    if df.empty:
        return "No GA4 data available."

    lines = [f"GA4 summary ({len(df)} rows):"]

    if "sessions" in df.columns:
        lines.append(f"  Total sessions: {df['sessions'].sum():,}")

    if "page_path" in df.columns and "sessions" in df.columns:
        top_pages = (
            df.groupby("page_path")["sessions"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        lines.append("  Top 5 pages by sessions:")
        for path, sessions in top_pages.items():
            lines.append(f"    {path}: {sessions:,}")

    if "bounce_rate" in df.columns:
        avg_bounce = df["bounce_rate"].mean()
        lines.append(f"  Avg bounce rate: {avg_bounce:.1%}")

    return "\n".join(lines)


def summarize_search_console(df: pd.DataFrame) -> str:
    """
    Return a text summary of Search Console data suitable for inclusion in a prompt.
    Expected columns: query, page, clicks, impressions, ctr, position
    Adjust column names to match your actual Search Console export schema.
    """
    if df.empty:
        return "No Search Console data available."

    lines = [f"Search Console summary ({len(df)} rows):"]

    if "clicks" in df.columns:
        lines.append(f"  Total clicks: {df['clicks'].sum():,}")
    if "impressions" in df.columns:
        lines.append(f"  Total impressions: {df['impressions'].sum():,}")

    if "query" in df.columns and "clicks" in df.columns:
        top_queries = (
            df.groupby("query")["clicks"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        lines.append("  Top 10 queries by clicks:")
        for query, clicks in top_queries.items():
            lines.append(f"    '{query}': {clicks:,} clicks")

    if "position" in df.columns and "query" in df.columns:
        # Queries ranking 4-10 with decent impressions — low-hanging fruit
        near_top = df[
            (df["position"] >= 4)
            & (df["position"] <= 10)
            & (df.get("impressions", pd.Series(dtype=float)) > 100)
        ]
        if not near_top.empty:
            lines.append(f"  Queries ranking 4-10 with >100 impressions: {len(near_top)}")

    return "\n".join(lines)


def build_analysis_context(ga4_df: pd.DataFrame, sc_df: pd.DataFrame, site_url: str) -> str:
    """Combine all data summaries into a single context string for the agent prompt."""
    return "\n\n".join([
        f"Site: {site_url}",
        summarize_ga4(ga4_df),
        summarize_search_console(sc_df),
    ])
