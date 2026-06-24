"""
Tool definitions in JSON Schema format.
These describe tools to the LLM — they do NOT implement the tools.
Implementations live in agent/tools/.

To port to Bedrock: the schema format is identical (JSON Schema).
Only the wrapper key changes: Gemini uses `function_declarations`,
Bedrock Converse uses `toolSpec` inside a `tools` list.
"""


def get_tool_definitions() -> list[dict]:
    """Return JSON Schema function declarations for all agent tools."""
    return [
        {
            "name": "fetch_page_metadata",
            "description": (
                "Fetch the HTML metadata for a URL: title tag, meta description, "
                "canonical URL, and H1/H2 headings. Use this to audit a page's SEO "
                "metadata without loading full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch metadata from."},
                },
                "required": ["url"],
            },
        },
        {
            "name": "fetch_page_text",
            "description": (
                "Fetch and extract the main visible text content from a URL. "
                "Use this to review page content for relevance, keyword usage, and quality."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch text from."},
                },
                "required": ["url"],
            },
        },
        {
            "name": "get_memory",
            "description": (
                "Read the agent's memory: last run timestamp, last recommendations made, "
                "and any previously analyzed pages. Use this at the start of each run "
                "to avoid repeating work."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "save_memory",
            "description": (
                "Save the agent's findings and recommendations to memory so future "
                "runs can build on this work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "string",
                        "description": "The recommendations generated this run.",
                    },
                    "pages_analyzed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs analyzed this run.",
                    },
                },
                "required": ["recommendations", "pages_analyzed"],
            },
        },
        {
            "name": "load_analytics_data",
            "description": (
                "Load GA4 and Search Console data from local sample files and return "
                "a structured summary. Use this to identify which pages and queries "
                "deserve attention."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]


def get_gemini_tool_config(tool_definitions: list[dict]):
    """Wrap tool definitions in the Gemini types.Tool format."""
    from google.genai import types
    return types.Tool(function_declarations=tool_definitions)
