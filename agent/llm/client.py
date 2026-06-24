"""
LLM client factory.
Returns a configured Gemini client today.
To swap to Bedrock: add a provider check and return a boto3 bedrock-runtime client instead.
"""

import os


def get_client(provider: str = "gemini"):
    """Return a configured LLM client for the given provider."""
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    raise NotImplementedError(
        f"Provider '{provider}' is not supported. "
        "Supported providers: 'gemini'. To add Bedrock, return a boto3 bedrock-runtime client here."
    )
