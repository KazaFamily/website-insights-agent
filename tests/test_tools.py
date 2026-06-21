"""
Unit tests for tool functions.
Uses sample DataFrames — no S3 or Gemini calls.
"""

import pandas as pd
import pytest

from agent.tools.analyzer import (
    summarize_ga4_pages,
    summarize_ga4_traffic,
    summarize_search_console_pages,
    summarize_search_console_queries,
    build_analysis_context,
)
from agent.tools.email_sender import format_email_body


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