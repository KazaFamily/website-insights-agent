"""
Unit tests for tool functions.
Uses sample DataFrames — no S3 or Gemini calls.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import urllib.error

from agent.tools.analyzer import (
    build_analysis_context,
    summarize_ga4_pages,
    summarize_ga4_traffic,
    summarize_search_console_pages,
    summarize_search_console_queries,
)
from agent.tools.email_sender import format_email_body
from agent.tools.memory import get_memory, save_memory
from agent.tools.page_fetcher import fetch_page_metadata, fetch_page_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ga4_pages():
    return pd.DataFrame({
        "Page title and screen class": ["Home", "Blog Post", "About"],
        "Views": [500, 200, 50],
        "Active users": [300, 100, 30],
        "Views per active user": [1.6, 2.0, 1.6],
        "Average engagement time per active user": [3.0, 120.0, 45.0],
        "Event count": [1000, 400, 100],
        "Key events": [0, 0, 0],
        "Total revenue": [0, 0, 0],
    })


@pytest.fixture
def sample_ga4_traffic():
    return pd.DataFrame({
        "Session source / medium": ["google / organic", "(direct) / (none)", "linkedin.com / referral"],
        "Sessions": [200, 150, 50],
        "Engaged sessions": [180, 100, 40],
        "Engagement rate": [0.90, 0.67, 0.80],
        "Average engagement time per session": [45.0, 25.0, 30.0],
        "Events per session": [10.0, 8.0, 6.0],
        "Event count": [2000, 1200, 300],
        "Key events": [20, 1, 5],
        "Session key event rate": [0.10, 0.007, 0.10],
        "Total revenue": [0, 0, 0],
    })


@pytest.fixture
def sample_sc_pages():
    return pd.DataFrame({
        "Top pages": ["https://example.com/", "https://example.com/blog", "https://example.com/about"],
        "Clicks": [50, 30, 5],
        "Impressions": [1000, 3000, 200],
        "CTR": ["5.00%", "1.00%", "2.50%"],
        "Position": [3.0, 9.5, 6.0],
    })


@pytest.fixture
def sample_sc_queries():
    return pd.DataFrame({
        "Top queries": ["best widget", "widget reviews", "buy widget", "widget guide"],
        "Clicks": [40, 20, 10, 0],
        "Impressions": [800, 600, 300, 150],
        "CTR": ["5.00%", "3.33%", "3.33%", "0.00%"],
        "Position": [2.5, 5.0, 8.0, 12.0],
    })


# ── GA4 Pages ─────────────────────────────────────────────────────────────────

def test_summarize_ga4_pages_empty():
    result = summarize_ga4_pages(pd.DataFrame())
    assert "No GA4 pages data" in result


def test_summarize_ga4_pages_total_views(sample_ga4_pages):
    result = summarize_ga4_pages(sample_ga4_pages)
    assert "750" in result  # 500 + 200 + 50


def test_summarize_ga4_pages_top_pages(sample_ga4_pages):
    result = summarize_ga4_pages(sample_ga4_pages)
    assert "Home" in result


def test_summarize_ga4_pages_high_engagement(sample_ga4_pages):
    result = summarize_ga4_pages(sample_ga4_pages)
    assert "Blog Post" in result  # 120s avg engagement


# ── GA4 Traffic ───────────────────────────────────────────────────────────────

def test_summarize_ga4_traffic_empty():
    result = summarize_ga4_traffic(pd.DataFrame())
    assert "No GA4 traffic data" in result


def test_summarize_ga4_traffic_total_sessions(sample_ga4_traffic):
    result = summarize_ga4_traffic(sample_ga4_traffic)
    assert "400" in result  # 200 + 150 + 50


def test_summarize_ga4_traffic_top_source(sample_ga4_traffic):
    result = summarize_ga4_traffic(sample_ga4_traffic)
    assert "google / organic" in result


# ── Search Console Pages ──────────────────────────────────────────────────────

def test_summarize_sc_pages_empty():
    result = summarize_search_console_pages(pd.DataFrame())
    assert "No Search Console pages data" in result


def test_summarize_sc_pages_total_clicks(sample_sc_pages):
    result = summarize_search_console_pages(sample_sc_pages)
    assert "85" in result  # 50 + 30 + 5


def test_summarize_sc_pages_low_ctr_opportunity(sample_sc_pages):
    result = summarize_search_console_pages(sample_sc_pages)
    assert "example.com/blog" in result  # 3000 impressions, 1% CTR


# ── Search Console Queries ────────────────────────────────────────────────────

def test_summarize_sc_queries_empty():
    result = summarize_search_console_queries(pd.DataFrame())
    assert "No Search Console queries data" in result


def test_summarize_sc_queries_total_clicks(sample_sc_queries):
    result = summarize_search_console_queries(sample_sc_queries)
    assert "70" in result  # 40 + 20 + 10 + 0


def test_summarize_sc_queries_top_query(sample_sc_queries):
    result = summarize_search_console_queries(sample_sc_queries)
    assert "best widget" in result


def test_summarize_sc_queries_near_top(sample_sc_queries):
    result = summarize_search_console_queries(sample_sc_queries)
    assert "widget guide" in result  # pos 12, 150 impressions, 0 clicks


# ── Context Builder ───────────────────────────────────────────────────────────

def test_build_analysis_context_includes_site_url(
    sample_ga4_pages, sample_ga4_traffic, sample_sc_pages, sample_sc_queries
):
    ctx = build_analysis_context(
        ga4_pages_df=sample_ga4_pages,
        ga4_traffic_df=sample_ga4_traffic,
        sc_pages_df=sample_sc_pages,
        sc_queries_df=sample_sc_queries,
        site_url="https://example.com",
    )
    assert "https://example.com" in ctx


def test_build_analysis_context_contains_all_sections(
    sample_ga4_pages, sample_ga4_traffic, sample_sc_pages, sample_sc_queries
):
    ctx = build_analysis_context(
        ga4_pages_df=sample_ga4_pages,
        ga4_traffic_df=sample_ga4_traffic,
        sc_pages_df=sample_sc_pages,
        sc_queries_df=sample_sc_queries,
        site_url="https://example.com",
    )
    assert "GA4 Pages" in ctx
    assert "GA4 Traffic" in ctx
    assert "Search Console — Pages" in ctx
    assert "Search Console — Queries" in ctx


# ── Email Formatter ───────────────────────────────────────────────────────────

def test_format_email_body_includes_site_url():
    body = format_email_body("https://example.com", "Some recommendations.")
    assert "https://example.com" in body
    assert "Some recommendations." in body


# ── Page Fetcher — live tests against sri-kaza.com/book ──────────────────────
#
# These hit the real site. The bot User-Agent triggers the pre-renderer so we
# get actual HTML rather than the empty React shell.
#
# Known issues surfaced by these tests (as of 2026-06-22):
#   - canonical points to homepage instead of self
#   - meta_description is the generic homepage blurb, not page-specific

LIVE_URL = "https://sri-kaza.com/book"


@pytest.mark.live
def test_live_fetch_page_metadata_title():
    result = fetch_page_metadata(LIVE_URL)
    assert "error" not in result
    assert result["title"] == (
        "Unconvention: A Small Business Strategy Guide | Award Winning, #1 Amazon Bestseller"
    )


@pytest.mark.live
def test_live_fetch_page_metadata_canonical():
    """Canonical points to homepage root — needs to be fixed in the pre-rendered HTML on the server."""
    result = fetch_page_metadata(LIVE_URL)
    assert result["canonical"] == "https://sri-kaza.com"  # TODO: should be LIVE_URL once server is fixed


@pytest.mark.live
def test_live_fetch_page_metadata_h1():
    result = fetch_page_metadata(LIVE_URL)
    assert result["h1s"] == ["The Strategy Guide Built for Small Business Owners"]


@pytest.mark.live
def test_live_fetch_page_metadata_h2s_present():
    result = fetch_page_metadata(LIVE_URL)
    assert len(result["h2s"]) >= 4
    h2_text = " ".join(result["h2s"])
    assert "Underdog" in h2_text


@pytest.mark.live
def test_live_fetch_page_text_truncated():
    result = fetch_page_text(LIVE_URL)
    assert "error" not in result
    assert result["truncated"] is True
    assert len(result["text"]) == 3000


@pytest.mark.live
def test_live_fetch_page_text_contains_expected_content():
    result = fetch_page_text(LIVE_URL)
    assert "Unconvention" in result["text"]
    assert "small business" in result["text"].lower()


# ── Page Fetcher — mocked unit tests ─────────────────────────────────────────

_SAMPLE_HTML = """
<html>
<head>
  <title>My Page Title</title>
  <meta name="description" content="A great page description.">
  <link rel="canonical" href="https://example.com/page">
</head>
<body>
  <h1>Main Heading</h1>
  <h2>Sub Heading One</h2>
  <h2>Sub Heading Two</h2>
  <p>Some body text here.</p>
</body>
</html>
"""


def _mock_response(html: str):
    mock = MagicMock()
    mock.read.return_value = html.encode("utf-8")
    mock.headers.get_content_charset.return_value = "utf-8"
    return mock


@patch("agent.tools.page_fetcher.urllib.request.urlopen")
def test_fetch_page_metadata_returns_expected_keys(mock_urlopen):
    mock_urlopen.return_value = _mock_response(_SAMPLE_HTML)
    result = fetch_page_metadata("https://example.com/page")
    assert result["url"] == "https://example.com/page"
    assert result["title"] == "My Page Title"
    assert result["meta_description"] == "A great page description."
    assert result["canonical"] == "https://example.com/page"
    assert "Main Heading" in result["h1s"]
    assert len(result["h2s"]) == 2


@patch("agent.tools.page_fetcher.urllib.request.urlopen")
def test_fetch_page_metadata_handles_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("connection refused")
    result = fetch_page_metadata("https://example.com/page")
    assert "error" in result
    assert result["url"] == "https://example.com/page"


@patch("agent.tools.page_fetcher.urllib.request.urlopen")
def test_fetch_page_text_truncates_long_content(mock_urlopen):
    long_body = "<p>" + ("x" * 4000) + "</p>"
    mock_urlopen.return_value = _mock_response(f"<html><body>{long_body}</body></html>")
    result = fetch_page_text("https://example.com/long")
    assert result["truncated"] is True
    assert len(result["text"]) == 3000


@patch("agent.tools.page_fetcher.urllib.request.urlopen")
def test_fetch_page_text_handles_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("timeout")
    result = fetch_page_text("https://example.com/page")
    assert "error" in result
    assert result["url"] == "https://example.com/page"


# ── Memory ────────────────────────────────────────────────────────────────────

def test_save_and_get_memory_roundtrip(tmp_path):
    memory_dir = str(tmp_path)
    save_memory(
        recommendations="Improve your title tags.",
        pages_analyzed=["https://example.com/", "https://example.com/blog"],
        memory_dir=memory_dir,
    )
    mem = get_memory(memory_dir=memory_dir)
    assert mem["last_run"]["recommendations"] == "Improve your title tags."
    assert "https://example.com/" in mem["last_run"]["pages_analyzed"]
    assert "https://example.com/blog" in mem["page_analyses"]


def test_get_memory_returns_defaults_when_missing(tmp_path):
    mem = get_memory(memory_dir=str(tmp_path))
    assert mem["last_run"] == {}
    assert mem["page_analyses"] == {}
