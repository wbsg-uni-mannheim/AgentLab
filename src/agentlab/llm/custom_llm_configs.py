from agentlab.llm.chat_api import OpenAIModelArgs

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
}