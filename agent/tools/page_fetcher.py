"""
Fetches and parses web pages.
Two tools:
  - fetch_page_metadata: title, meta description, canonical, H1s, H2s
  - fetch_page_text: cleaned visible body text
"""

import urllib.request
from bs4 import BeautifulSoup


def _get(url: str):
    """Make a bare HTTP request sending only the User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": "WebsiteInsightsBot/1.0"})
    return urllib.request.urlopen(req, timeout=10)


def _read(response) -> str:
    raw = response.read()
    charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def fetch_page_metadata(url: str) -> dict:
    """Return title, meta description, canonical URL, H1s, and H2s for a page."""
    try:
        soup = BeautifulSoup(_read(_get(url)), "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = desc_tag.get("content") if desc_tag else None

        canonical_tag = soup.find("link", attrs={"rel": "canonical"})
        canonical = canonical_tag.get("href") if canonical_tag else None

        h1s = [tag.get_text(strip=True) for tag in soup.find_all("h1")]
        h2s = [tag.get_text(strip=True) for tag in soup.find_all("h2")]

        return {
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "canonical": canonical,
            "h1s": h1s,
            "h2s": h2s,
        }
    except Exception as e:
        return {"url": url, "error": str(e)}


def fetch_page_text(url: str) -> dict:
    """Return cleaned visible body text from a page, truncated to 3000 chars."""
    try:
        soup = BeautifulSoup(_read(_get(url)), "html.parser")

        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        truncated = len(text) > 3000
        return {"url": url, "text": text[:3000], "truncated": truncated}
    except Exception as e:
        return {"url": url, "error": str(e)}
