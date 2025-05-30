from agentlab.llm.chat_api import OpenAIModelArgs, AnthropicModelArgs

# Custom LLM configurations
# This file contains custom model configurations that won't conflict with upstream updates
CUSTOM_CHAT_MODEL_ARGS_DICT = {
    "openai/gpt-4.1-2025-04-14": OpenAIModelArgs(
        model_name="gpt-4.1-2025-04-14",
        max_total_tokens=1_047_576,
        max_input_tokens=1_047_576,
        max_new_tokens=32_768,
        vision_support=True,
    ),
    "anthropic/claude-sonnet-4-20250514": AnthropicModelArgs(
        model_name="claude-sonnet-4-20250514",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=64_000,
        vision_support=True,
        temperature=0.7,
    ),
}