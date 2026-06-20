"""
Unit tests for tool functions.
Uses sample CSVs from data/sample/ — no S3 or Gemini calls.
"""

import pandas as pd
import pytest

from agent.tools.analyzer import summarize_ga4, summarize_search_console, build_analysis_context
from agent.tools.email_sender import format_email_body


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ga4():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "page_path": ["/home", "/blog/post-1"],
        "sessions": [500, 200],
        "bounce_rate": [0.45, 0.60],
    })


@pytest.fixture
def sample_sc():
    return pd.DataFrame({
        "query": ["best widget", "widget reviews", "buy widget"],
        "page": ["/widgets", "/reviews", "/shop"],
        "clicks": [120, 80, 40],
        "impressions": [2000, 1500, 800],
        "ctr": [0.06, 0.053, 0.05],
        "position": [3.2, 5.1, 7.8],
    })


# ── GA4 summarizer ────────────────────────────────────────────────────────────

def test_summarize_ga4_empty():
    result = summarize_ga4(pd.DataFrame())
    assert "No GA4 data" in result


def test_summarize_ga4_total_sessions(sample_ga4):
    result = summarize_ga4(sample_ga4)
    assert "700" in result  # 500 + 200


def test_summarize_ga4_top_pages(sample_ga4):
    result = summarize_ga4(sample_ga4)
    assert "/home" in result


# ── Search Console summarizer ─────────────────────────────────────────────────

def test_summarize_sc_empty():
    result = summarize_search_console(pd.DataFrame())
    assert "No Search Console data" in result


def test_summarize_sc_total_clicks(sample_sc):
    result = summarize_search_console(sample_sc)
    assert "240" in result  # 120 + 80 + 40


def test_summarize_sc_top_queries(sample_sc):
    result = summarize_search_console(sample_sc)
    assert "best widget" in result


# ── Context builder ───────────────────────────────────────────────────────────

def test_build_analysis_context_includes_site_url(sample_ga4, sample_sc):
    ctx = build_analysis_context(sample_ga4, sample_sc, "https://example.com")
    assert "https://example.com" in ctx


# ── Email formatter ───────────────────────────────────────────────────────────

def test_format_email_body_includes_site_url():
    body = format_email_body("https://example.com", "Some recommendations.")
    assert "https://example.com" in body
    assert "Some recommendations." in body
