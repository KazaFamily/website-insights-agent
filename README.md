# Website Insights Agent

An AI agent that reads website analytics data from S3, generates actionable recommendations using Gemini, and delivers them via email through AWS Lambda/SES.

Built as a capstone project for the [AI Agents: Intensive Vibe Coding Course](https://www.kaggle.com/learn/ai-agents) with Google/Kaggle.

## Architecture

```
S3 (GA4 + Search Console CSVs)
        │
        ▼
  agent/tools/s3_reader.py   ← reads raw data
        │
        ▼
  agent/tools/analyzer.py    ← summarizes into prompt context
        │
        ▼
  Gemini API (raw call)      ← generates recommendations
        │
        ▼
  agent/tools/email_sender.py ← POSTs to Lambda → SES → inbox
```

**Design principles:**
- Raw Gemini API calls — no ADK, no LangGraph (easy to port to Bedrock)
- Tools are plain Python functions
- Config-driven via `config.yaml`
- Designed to add sub-agents later without restructuring

## Setup

```bash
pip install -r requirements.txt
cp config.yaml config.local.yaml  # edit with your values
export GEMINI_API_KEY=your_key_here
```

AWS credentials should be available via environment variables or IAM role:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

## Running

```bash
python -m agent.agent
```

## Testing

```bash
pytest tests/
```

## Project Structure

```
website-insights-agent/
├── config.yaml              # site URL, S3 paths, model, email config
├── requirements.txt
├── agent/
│   ├── agent.py             # main agent loop
│   └── tools/
│       ├── s3_reader.py     # reads GA4 + Search Console from S3
│       ├── analyzer.py      # summarizes data into prompt context
│       └── email_sender.py  # sends recommendations via Lambda
├── data/sample/             # sample CSVs for local testing
├── notebooks/
│   └── capstone_demo.ipynb  # Kaggle submission notebook
└── tests/
    └── test_tools.py
```

## Data Sources

- **GA4**: CSV exports placed in `s3://{bucket}/{ga4_prefix}/`
- **Search Console**: CSV exports placed in `s3://{bucket}/{search_console_prefix}/`

Expected GA4 columns: `date, page_path, sessions, bounce_rate, avg_session_duration`  
Expected Search Console columns: `query, page, clicks, impressions, ctr, position`
