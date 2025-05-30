# Anthropic pricing support using langchain_community
import logging

def get_pricing_anthropic():
    """Get pricing information for Anthropic models from langchain_community.
    Falls back to hardcoded values if langchain_community is not available.
    Returns a dictionary with model names as keys and pricing info as values.
    """

    # Fallback to hardcoded pricing (as of May 2025)
    return {
        "claude-sonnet-4-20250514": {
            "prompt": 3.00 / 1_000_000,  # $0.25 per million input tokens
            "completion": 15.00 / 1_000_000,  # $1.25 per million output tokens
        },
    }