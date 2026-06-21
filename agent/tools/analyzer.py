"""
Summarizes Two GA4 and Two Search Console DataFrames into plain-text 
context that the agent passes to an LLM for recommendation generation.
"""

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_ctr(df: pd.DataFrame, col: str = "CTR") -> pd.DataFrame:
    """Convert CTR from '4.92%' string to float 4.92 (percentage points)."""
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
    return df


# ── GA4: Pages & Screens ──────────────────────────────────────────────────────

def summarize_ga4_pages(df: pd.DataFrame, top_n: int = 10) -> str:
    """
    Summarize the GA4 'Pages and screens' export.

    Expected columns (after skiprows=9):
        Page title and screen class, Views, Active users,
        Views per active user, Average engagement time per active user,
        Event count, Key events, Total revenue
    """
    if df.empty:
        return "No GA4 pages data available."

    lines = [f"GA4 Pages & Screens ({len(df)} pages, last 90 days):"]

    # Total views
    if "Views" in df.columns:
        lines.append(f"  Total views: {df['Views'].sum():,}")

    # Total active users (deduplicated estimate)
    if "Active users" in df.columns:
        lines.append(f"  Total active users: {df['Active users'].sum():,}")

    # Top pages by views
    if "Page title and screen class" in df.columns and "Views" in df.columns:
        top = (
            df.groupby("Page title and screen class")["Views"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        lines.append(f"\n  Top {top_n} pages by views:")
        for title, views in top.items():
            lines.append(f"    {title[:80]}: {views:,} views")

    # High-engagement pages (avg engagement time > 60s, at least 5 active users)
    eng_col = "Average engagement time per active user"
    if eng_col in df.columns and "Active users" in df.columns:
        engaged = df[
            (df[eng_col] > 60) & (df["Active users"] >= 5)
        ].sort_values(eng_col, ascending=False).head(5)
        if not engaged.empty:
            lines.append("\n  High-engagement pages (>60s avg, 5+ users):")
            for _, row in engaged.iterrows():
                title = str(row["Page title and screen class"])[:70]
                lines.append(
                    f"    {title}: {row[eng_col]:.0f}s avg, "
                    f"{row['Active users']:.0f} users"
                )

    # Low-engagement pages with decent traffic (potential content issues)
    if eng_col in df.columns and "Views" in df.columns:
        low_eng = df[
            (df[eng_col] < 15) & (df["Views"] >= 20)
        ].sort_values("Views", ascending=False).head(5)
        if not low_eng.empty:
            lines.append("\n  Low-engagement pages (<15s avg, 20+ views) — possible issues:")
            for _, row in low_eng.iterrows():
                title = str(row["Page title and screen class"])[:70]
                lines.append(
                    f"    {title}: {row[eng_col]:.0f}s avg, "
                    f"{row['Views']:.0f} views"
                )

    return "\n".join(lines)


# ── GA4: Traffic Acquisition ──────────────────────────────────────────────────

def summarize_ga4_traffic(df: pd.DataFrame) -> str:
    """
    Summarize the GA4 'Traffic acquisition' export.

    Expected columns (after skiprows=9):
        Session source / medium, Sessions, Engaged sessions,
        Engagement rate, Average engagement time per session,
        Events per session, Event count, Key events,
        Session key event rate, Total revenue
    """
    if df.empty:
        return "No GA4 traffic data available."

    lines = [f"GA4 Traffic Acquisition ({len(df)} sources, last 90 days):"]

    source_col = "Session source / medium"

    if "Sessions" in df.columns:
        total = df["Sessions"].sum()
        lines.append(f"  Total sessions: {total:,}")

        # Sessions by source
        top_sources = df.sort_values("Sessions", ascending=False).head(8)
        lines.append("\n  Sessions by source:")
        for _, row in top_sources.iterrows():
            pct = (row["Sessions"] / total * 100) if total > 0 else 0
            eng = f", engagement rate: {row['Engagement rate']:.0%}" if "Engagement rate" in df.columns else ""
            key_events = f", key events: {int(row['Key events'])}" if "Key events" in df.columns else ""
            lines.append(
                f"    {row[source_col]}: {int(row['Sessions']):,} sessions "
                f"({pct:.1f}%){eng}{key_events}"
            )

    # Best converting sources by key events (exclude very low traffic)
    if "Key events" in df.columns and "Sessions" in df.columns:
        converters = df[df["Sessions"] >= 5].sort_values(
            "Session key event rate", ascending=False
        ).head(3)
        if not converters.empty and converters["Key events"].sum() > 0:
            lines.append("\n  Best converting sources (by key event rate):")
            for _, row in converters.iterrows():
                if row["Key events"] > 0:
                    lines.append(
                        f"    {row[source_col]}: "
                        f"{row['Session key event rate']:.1%} key event rate, "
                        f"{int(row['Key events'])} events"
                    )

    return "\n".join(lines)


# ── Search Console: Pages ─────────────────────────────────────────────────────

def summarize_search_console_pages(df: pd.DataFrame) -> str:
    """
    Summarize the Search Console pages export.

    Expected columns:
        Top pages, Clicks, Impressions, CTR, Position

    CTR is cleaned from '4.92%' string format automatically.
    """
    if df.empty:
        return "No Search Console pages data available."

    df = _clean_ctr(df)
    lines = [f"Search Console — Pages ({len(df)} pages, last 90 days):"]

    if "Clicks" in df.columns:
        lines.append(f"  Total clicks: {df['Clicks'].sum():,}")
    if "Impressions" in df.columns:
        lines.append(f"  Total impressions: {df['Impressions'].sum():,}")

    # Top pages by clicks
    if "Clicks" in df.columns:
        top = df.sort_values("Clicks", ascending=False).head(5)
        lines.append("\n  Top 5 pages by clicks:")
        for _, row in top.iterrows():
            lines.append(
                f"    {row['Top pages']}: {int(row['Clicks'])} clicks, "
                f"{int(row['Impressions'])} impressions, "
                f"pos {row['Position']:.1f}"
            )

    # High impression / low CTR — biggest content/title opportunities
    if "Impressions" in df.columns and "CTR" in df.columns:
        opportunities = df[
            (df["Impressions"] > 200) & (df["CTR"] < 2.0)
        ].sort_values("Impressions", ascending=False).head(5)
        if not opportunities.empty:
            lines.append(
                "\n  High-impression / low-CTR pages — title or meta description opportunities:"
            )
            for _, row in opportunities.iterrows():
                lines.append(
                    f"    {row['Top pages']}: "
                    f"{int(row['Impressions'])} impressions, "
                    f"{row['CTR']:.2f}% CTR, "
                    f"pos {row['Position']:.1f}"
                )

    return "\n".join(lines)


# ── Search Console: Queries ───────────────────────────────────────────────────

def summarize_search_console_queries(df: pd.DataFrame) -> str:
    """
    Summarize the Search Console queries export.

    Expected columns:
        Top queries, Clicks, Impressions, CTR, Position

    CTR is cleaned from '4.92%' string format automatically.
    """
    if df.empty:
        return "No Search Console queries data available."

    df = _clean_ctr(df)
    lines = [f"Search Console — Queries ({len(df)} queries, last 90 days):"]

    if "Clicks" in df.columns:
        lines.append(f"  Total clicks: {df['Clicks'].sum():,}")

    # Top queries by clicks
    top = df.sort_values("Clicks", ascending=False).head(10)
    lines.append("\n  Top 10 queries by clicks:")
    for _, row in top.iterrows():
        lines.append(
            f"    '{row['Top queries']}': {int(row['Clicks'])} clicks, "
            f"{int(row['Impressions'])} impressions, "
            f"pos {row['Position']:.1f}"
        )

    # Near-top queries: ranking 5-30, impressions > 20, low/no clicks
    # These are the best content improvement opportunities
    if "Position" in df.columns and "Impressions" in df.columns:
        near_top = df[
            (df["Position"] >= 5)
            & (df["Position"] <= 30)
            & (df["Impressions"] > 20)
            & (df["Clicks"] == 0)
        ].sort_values("Impressions", ascending=False).head(8)
        if not near_top.empty:
            lines.append(
                "\n  Ranking but not clicking — content/title optimization opportunities:"
            )
            for _, row in near_top.iterrows():
                lines.append(
                    f"    '{row['Top queries']}': "
                    f"{int(row['Impressions'])} impressions, "
                    f"0 clicks, pos {row['Position']:.1f}"
                )

    return "\n".join(lines)


# ── Context Builder ───────────────────────────────────────────────────────────

def build_analysis_context(
    ga4_pages_df: pd.DataFrame,
    ga4_traffic_df: pd.DataFrame,
    sc_pages_df: pd.DataFrame,
    sc_queries_df: pd.DataFrame,
    site_url: str,
) -> str:
    """
    Combine all four data summaries into a single context string
    for the agent's Gemini prompt.
    """
    return "\n\n".join([
        f"Site: {site_url}",
        summarize_ga4_pages(ga4_pages_df),
        summarize_ga4_traffic(ga4_traffic_df),
        summarize_search_console_pages(sc_pages_df),
        summarize_search_console_queries(sc_queries_df),
    ])